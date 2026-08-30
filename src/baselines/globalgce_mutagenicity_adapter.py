"""Train-only GlobalGCE adaptation for the strict Mutagenicity protocol.

The project-owned layer controls cohort membership, provenance, chemistry
conversion, RF target validation, resumability, and audit. The actual rule
mining, rule optimization, and rule-to-counterfactual application are delegated
to the official GlobalGCE implementation.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import io
import json
import math
import os
import random
import resource
import site
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from src.data.dataset_registry import get_dataset_spec

from .globalgce_bace_native_rules import OFFICIAL_GLOBALGCE_COMMIT

from .globalgce_resumable import (
    GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION,
    _normalize_checkpoint_file_evidence,
    normalize_globalgce_training_resume_identity,
    train_globalgce_resumable,
    validate_exact_top_k_proof_identity,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - runtime dependency
    Chem = None

SOURCE_LABEL = 1
TARGET_LABEL = 0
GENERATOR_METHOD = "GlobalGCE"
DATASET_NAME = "Mutagenicity"
# Raw TU labels from Mutagenicity_label_readme.txt. Official preprocessing
# shifts both node and edge labels by +1 so internal class 0 stays padding.
OFFICIAL_MUTAGENICITY_NODE_LABEL_TO_SYMBOL = {
    0: "C",
    1: "O",
    2: "Cl",
    3: "H",
    4: "N",
    5: "F",
    6: "Br",
    7: "S",
    8: "P",
    9: "I",
    10: "Na",
    11: "K",
    12: "Li",
    13: "Ca",
}
OFFICIAL_MUTAGENICITY_EDGE_LABEL_TO_BOND = {
    0: "single",
    1: "double",
    2: "triple",
}
TASTE_EXACT_ATOM_VOCABULARY_SOURCE = (
    "taste_train_observed_exact_symbols_frozen_gine_atomic_num_v1"
)
DEFAULT_EXPECTED_PARENT_COUNT = 1448
DEFAULT_NATIVE_TRAIN_CSV = (
    "outputs/hpc/datasets/final/mutagenicity_v1_processed/train.csv"
)
REQUIRED_OUTPUT_FILES = (
    "raw_generated_candidates.jsonl",
    "candidate_pool.jsonl",
    "candidate_universe.jsonl",
    "invalid_candidates.jsonl",
    "non_target_candidates.jsonl",
    "training_summary.json",
    "summary.json",
    "run_manifest.json",
    "resume_checkpoint.json",
    "generation_resume_checkpoint.json",
    "_RUN_COMPLETE.json",
)
FROZEN_GINE_IN_MEMORY_FILES = frozenset(
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
OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA = "official_globalgce_api_signature_v1"
OFFICIAL_GLOBALGCE_MODULE_PROVENANCE_SCHEMA = (
    "official_globalgce_module_provenance_v1"
)
OFFICIAL_API_SIGNATURE_FILE = "official_api_signature.json"
PYTHON_MODULE_PROVENANCE_FILE = "python_module_provenance.json"
PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES = {
    "models.GTGNN.GTGNN": (
        "(x_dim, h_dim, n_out, num_edge_attr, device, save_model_path)"
    ),
    "models.GlobalGCE.GlobalGCE": (
        "(x_dim, h_dim, z_dim, edge_attr_dim, dropout, fs_min_nodes, "
        "fs_max_nodes, topk_fs, random_subgraph, save_fs_path, device, gt_gnn)"
    ),
    "models.GlobalGCE.GlobalGCE.get_fs_expanded_data": (
        "(self, train_loader, crop_expansion=False)"
    ),
    "models.GlobalGCE.GlobalGCE.get_rules": "(self, fss)",
    "models.GlobalGCE.GlobalGCE.run_one_batch": "(self, rules, data)",
    "models.GlobalGCE.generate_cfs": "(dataloader, rules, device)",
    "models.GlobalGCE.concate_inputs_with_local_recourse": (
        "(features, adj, edge_attrs, fs_index_list, mask_idx_list, rules, device)"
    ),
    "models.fsg.FrequentSubgraphGenerator": (
        "(fs_min_vertices, fs_max_vertices, save_fs_path, topk, "
        "random_subgraph=False)"
    ),
    "models.gSpan.gSpan.gSpan": (
        "(nx_graph_list, min_support=10, min_num_vertices=1, "
        "max_num_vertices=inf, max_ngraphs=inf, is_undirected=True, "
        "verbose=False, visualize=False, where=False)"
    ),
    "models.models_utils.train_globalgce": (
        "(epochs, pred_model, model, lr, train_loader, val_loader, "
        "save_rule_path, save_model_path)"
    ),
    "models.models_utils.test_globalgce": (
        "(data_loader, model, gnn_model, rules)"
    ),
}


class GlobalGCEMutagenicityCodecError(RuntimeError):
    """Raised when the native source graph codec fails its round-trip gate."""


class GlobalGCEEmptyCandidateUniverseError(RuntimeError):
    """Raised when native generation yields no valid target candidates."""


@dataclass(frozen=True, slots=True)
class GlobalGCECodecMetadata:
    """Internal label metadata used by the current native GlobalGCE tensors."""

    atom_symbols: tuple[str, ...]
    bond_names: tuple[str, ...]
    formal_charge_encoded_by_native: bool = False
    atom_attribute_source: str = "source_anchored"
    source_atom_mapping_method: str = (
        "rdkit_atom_index_preserved_by_dense_builder"
    )
    atom_vocabulary_source: str = (
        "official_mutagenicity_train_observed_exact_symbols_v1"
    )
    atom_vocabulary_feature_schema_sha256: str | None = None

    @classmethod
    def from_dataset(cls, dataset: Any) -> "GlobalGCECodecMetadata":
        atom_symbols = tuple(str(value) for value in dataset.atom_symbols)
        bond_names = tuple(str(value) for value in dataset.bond_names)
        if not atom_symbols:
            raise GlobalGCEMutagenicityCodecError(
                "Native dataset has no atom label metadata."
            )
        if not bond_names or bond_names[0] not in {"padding", "no_edge"}:
            raise GlobalGCEMutagenicityCodecError(
                "Native dataset bond metadata must explicitly reserve internal "
                "label 0 for no-edge/padding."
            )
        return cls(
            atom_symbols=atom_symbols,
            bond_names=bond_names,
            formal_charge_encoded_by_native=bool(
                getattr(dataset, "formal_charge_encoded_by_native", False)
            ),
            atom_attribute_source=str(
                getattr(dataset, "atom_attribute_source", "source_anchored")
            ),
            source_atom_mapping_method=str(
                getattr(
                    dataset,
                    "source_atom_mapping_method",
                    "rdkit_atom_index_preserved_by_dense_builder",
                )
            ),
            atom_vocabulary_source=str(
                getattr(
                    dataset,
                    "atom_vocabulary_source",
                    "official_mutagenicity_train_observed_exact_symbols_v1",
                )
            ),
            atom_vocabulary_feature_schema_sha256=(
                None
                if getattr(
                    dataset,
                    "atom_vocabulary_feature_schema_sha256",
                    None,
                )
                is None
                else str(dataset.atom_vocabulary_feature_schema_sha256)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_label_mapping": {
                "0": "padding",
                **{
                    str(index + 1): symbol
                    for index, symbol in enumerate(self.atom_symbols)
                },
            },
            "edge_label_mapping": {
                str(index): ("no_edge" if index == 0 else name)
                for index, name in enumerate(self.bond_names)
            },
            "official_raw_node_label_mapping": {
                str(index): symbol
                for index, symbol in OFFICIAL_MUTAGENICITY_NODE_LABEL_TO_SYMBOL.items()
            },
            "official_raw_edge_label_mapping": {
                str(index): name
                for index, name in OFFICIAL_MUTAGENICITY_EDGE_LABEL_TO_BOND.items()
            },
            "formal_charge_encoded_by_native": (
                self.formal_charge_encoded_by_native
            ),
            "atom_vocabulary_source": self.atom_vocabulary_source,
            "atom_vocabulary_feature_schema_sha256": (
                self.atom_vocabulary_feature_schema_sha256
            ),
            "atom_attribute_source": self.atom_attribute_source,
            "source_atom_mapping_method": self.source_atom_mapping_method,
        }


@dataclass(slots=True)
class GlobalGCEGraphDecodeResult:
    ok: bool
    smiles: str | None
    mol: Any | None
    codec_decoded: bool
    error_type: str | None
    error_message: str | None
    num_atoms: int
    num_bonds: int
    graph_record: dict[str, Any]


class TeacherProtocol(Protocol):
    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


class NativeGeneratorProtocol(Protocol):
    def config_identity(self) -> dict[str, Any]:
        ...

    def generate(
        self,
        parents: Sequence["TrainParent"],
        *,
        output_dir: Path,
        seed: int,
        epochs: int,
        top_k_native: int,
        learning_rate: float,
        dropout: float,
        device: str,
        resume: bool,
        generation_chunk_size: int,
        generation_num_workers: int,
        memory_log_every_chunks: int,
        gspan_flush_every: int,
        gspan_max_in_memory_candidates: int,
        gspan_exact_top_k_pruning: bool,
        gspan_adoption_proof: str | Path | None,
        start_parent_offset: int,
        on_training_ready: Callable[[dict[str, Any]], None] | None,
        on_chunk: (
            Callable[[int, int, int, list[dict[str, Any]]], None] | None
        ),
        rules_only: bool = False,
        expected_resume_checkpoint: Mapping[str, Any] | None = None,
        on_resume_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        after_epoch_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        on_generation_complete: Callable[[], None] | None = None,
    ) -> "NativeGenerationResult":
        ...


@dataclass(frozen=True, slots=True)
class TrainParent:
    parent_id: str
    smiles: str
    label: int
    split: str


@dataclass(frozen=True, slots=True)
class PoolBuildConfig:
    parent_limit: int = 0
    expected_parent_count: int = DEFAULT_EXPECTED_PARENT_COUNT
    seed: int = 13
    epochs: int = 100
    top_k_native: int = 20
    learning_rate: float = 0.1
    dropout: float = 0.5
    device: str = "cuda"
    resume: bool = True
    forbid_calibration_test: bool = True
    generation_chunk_size: int = 32
    generation_num_workers: int = 0
    memory_log_every_chunks: int = 1
    gspan_flush_every: int = 256
    gspan_max_in_memory_candidates: int = 256
    gspan_exact_top_k_pruning: bool = False
    gspan_adoption_proof: str | None = None


@dataclass(slots=True)
class NativeGenerationResult:
    records: list[dict[str, Any]]
    training_summary: dict[str, Any]


class _DenseMoleculeDataset:
    """Small compatibility surface expected by official GlobalGCE classes."""

    def __init__(
        self,
        *,
        torch_module: Any,
        parent_ids: Sequence[str],
        feat: Any,
        adj: Any,
        edge_attr: Any,
        labels: Any,
        num_nodes: Any,
        num_edges: Any,
        train_idx: Sequence[int],
        val_idx: Sequence[int],
        test_idx: Sequence[int],
        atom_symbols: Sequence[str],
        bond_names: Sequence[str],
        source_atom_attributes: Sequence[Sequence[dict[str, Any]]],
        dataset_name: str = DATASET_NAME,
        num_classes: int = 2,
        atom_vocabulary_source: str = (
            "official_mutagenicity_train_observed_exact_symbols_v1"
        ),
        atom_vocabulary_feature_schema_sha256: str | None = None,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.parent_ids = list(parent_ids)
        self.feat = feat
        self.adj = adj
        self.edge_attr = edge_attr
        self.labels = labels
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.train_idx = list(train_idx)
        self.val_idx = list(val_idx)
        self.test_idx = list(test_idx)
        self.max_num_nodes = int(feat.shape[1])
        self.node_feat_dim = int(feat.shape[-1])
        self.edge_attr_dim = int(edge_attr.shape[-1]) if edge_attr is not None else 0
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("Dense GlobalGCE num_classes must be an exact integer >= 2.")
        self.num_classes = num_classes
        self.index = torch_module.arange(len(parent_ids), dtype=torch_module.long)
        self.atom_symbols = list(atom_symbols)
        self.bond_names = list(bond_names)
        self.source_atom_attributes = [
            [dict(atom) for atom in row] for row in source_atom_attributes
        ]
        if len(self.source_atom_attributes) != len(self.parent_ids):
            raise GlobalGCEMutagenicityCodecError(
                "Source atom sidecar row count does not match native dataset: "
                f"sidecars={len(self.source_atom_attributes)}, "
                f"parents={len(self.parent_ids)}."
            )
        self.formal_charge_encoded_by_native = False
        self.atom_attribute_source = "source_anchored"
        self.source_atom_mapping_method = (
            "rdkit_atom_index_preserved_by_dense_builder"
        )
        self.atom_vocabulary_source = str(atom_vocabulary_source)
        self.atom_vocabulary_feature_schema_sha256 = (
            None
            if atom_vocabulary_feature_schema_sha256 is None
            else str(atom_vocabulary_feature_schema_sha256)
        )

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {
            "adj": self.adj[index].clone(),
            "feature": self.feat[index].clone(),
            "label": self.labels[index].clone(),
            "max_num_nodes": self.max_num_nodes,
            "index": self.index[index].clone(),
            "num_nodes": self.num_nodes[index].clone(),
            "num_edges": self.num_edges[index].clone(),
        }
        if self.edge_attr is not None:
            row["edge_attr"] = self.edge_attr[index].clone()
        return row


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text_at_directory(
    directory_fd: int,
    name: str,
    text: str,
) -> None:
    """Atomically publish one direct child of a retained directory fd."""

    if (
        type(directory_fd) is not int
        or directory_fd < 0
        or type(name) is not str
        or not name
        or name in {".", ".."}
        or Path(name).name != name
    ):
        raise ValueError("Descriptor-authorized atomic target is malformed.")
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
        raise ValueError("Descriptor-authorized atomic parent is not a directory.")
    temporary = f".{name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
    try:
        payload = text.encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Descriptor-authorized atomic write made no progress.")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _direct_retained_directory_target(path: Path) -> tuple[int, str] | None:
    parts = path.parts
    if (
        sys.platform.startswith("linux")
        and len(parts) == 6
        and parts[0] == os.sep
        and parts[1:4] == ("proc", "self", "fd")
        and parts[4].isdigit()
    ):
        return int(parts[4]), parts[5]
    return None


def _atomic_write_text(path: Path, text: str) -> None:
    retained_target = _direct_retained_directory_target(path)
    if retained_target is not None:
        _atomic_write_text_at_directory(
            retained_target[0],
            retained_target[1],
            text,
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _source_atom_attribute_audit_path(output_dir: Path) -> Path:
    """Keep each branch audit under its retained, independently owned root."""

    return output_dir / "source_atom_attribute_audit.jsonl"


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _truncate_jsonl(path: Path, row_count: int) -> None:
    if int(row_count) < 0:
        raise ValueError("JSONL row_count must be non-negative.")
    if not path.exists():
        if int(row_count) != 0:
            raise ValueError(
                f"Resume JSONL is missing but checkpoint expects {row_count} rows: "
                f"{path}"
            )
        _write_jsonl(path, [])
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.resume.",
        suffix=".tmp",
        dir=path.parent,
    )
    copied = 0
    try:
        with path.open("r", encoding="utf-8") as source, os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                if copied >= int(row_count):
                    break
                target.write(line)
                copied += 1
            target.flush()
            os.fsync(target.fileno())
        if copied != int(row_count):
            raise ValueError(
                f"Resume JSONL row mismatch for {path}: "
                f"expected_at_least={row_count}, found={copied}."
            )
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def _rss_gb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0**2)
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maximum / (1024.0**3)
    return maximum / (1024.0**2)


def log_globalgce_phase_memory(
    *,
    phase: str,
    chunk_index: int,
    processed_parent_count: int,
    raw_generated_count: int,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    cuda_allocated = 0.0
    cuda_reserved = 0.0
    if (
        torch_module is not None
        and hasattr(torch_module, "cuda")
        and torch_module.cuda.is_available()
    ):
        cuda_allocated = float(torch_module.cuda.memory_allocated()) / (1024.0**3)
        cuda_reserved = float(torch_module.cuda.memory_reserved()) / (1024.0**3)
    payload = {
        "phase": str(phase),
        "chunk_index": int(chunk_index),
        "processed_parent_count": int(processed_parent_count),
        "raw_generated_count": int(raw_generated_count),
        "rss_gb": _rss_gb(),
        "cuda_allocated_gb": cuda_allocated,
        "cuda_reserved_gb": cuda_reserved,
    }
    print(
        "[GLOBALGCE_MEMORY] "
        + " ".join(f"{key}={value}" for key, value in payload.items()),
        flush=True,
    )
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_payload(payload: bytes) -> list[dict[str, str]]:
    if type(payload) is not bytes or not payload:
        raise ValueError("Descriptor-authorized native train CSV payload is empty.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Descriptor-authorized native train CSV payload is not UTF-8."
        ) from exc
    return [dict(row) for row in csv.DictReader(io.StringIO(text, newline=""))]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_path_or_resolve(
    path_like: str | Path,
    *,
    strict: bool = False,
) -> Path:
    """Preserve a Linux /proc/self/fd authority instead of resolving its link."""

    path = Path(path_like).expanduser()
    parts = path.parts
    if (
        sys.platform.startswith("linux")
        and len(parts) >= 5
        and parts[0] == os.sep
        and parts[1:4] == ("proc", "self", "fd")
        and parts[4].isdigit()
    ):
        if strict and not path.exists():
            raise FileNotFoundError(path)
        return path
    return path.resolve(strict=strict)


def _file_identity(path_like: str | Path) -> dict[str, Any]:
    path = _descriptor_path_or_resolve(path_like)
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if path.is_file():
        payload["sha256"] = _sha256_file(path)
    return payload


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def stable_candidate_id(
    canonical_smiles: str, *, dataset_name: str = DATASET_NAME
) -> str:
    digest = hashlib.sha256(str(canonical_smiles).encode("utf-8")).hexdigest()
    prefix = "BACE" if str(dataset_name).strip().lower() == "bace" else "MUT"
    return f"{prefix}_GLOBALGCE_{digest[:20].upper()}"


def _normalized_tokens(path: Path) -> set[str]:
    normalized = path.name.lower().replace("-", "_").replace(".", "_")
    return {token for token in normalized.split("_") if token}


def _reject_non_train_path(path: Path, *, description: str) -> None:
    name = path.name.lower()
    tokens = _normalized_tokens(path)
    if "calibration_source" in name or "test_source" in name:
        raise ValueError(f"{description} references a forbidden cohort: {path}")
    if "calibration" in tokens or "test" in tokens:
        raise ValueError(f"{description} is not a train-only path: {path}")


def _canonical_smiles(smiles: str) -> str | None:
    if Chem is None:
        raise RuntimeError("RDKit is required for GlobalGCE Mutagenicity export.")
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None or molecule.GetNumAtoms() <= 0:
        return None
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _exact_csv_class_integer(
    value: Any,
    *,
    path: Path,
    row_number: int,
    field: str,
) -> int:
    """Parse one canonical non-negative decimal class token.

    CSV class identities are not measurements.  Float coercion would silently
    map values such as ``0.9`` and ``1.5`` to valid classes, so accept only the
    native textual grammar emitted by the prepared split writers.
    """

    if (
        type(value) is not str
        or not value
        or any(character not in "0123456789" for character in value)
        or (len(value) > 1 and value[0] == "0")
    ):
        raise ValueError(
            f"Invalid exact integer {field} at {path}:{row_number}: {value!r}"
        )
    return int(value)


def load_strict_train_parents(
    train_csv: str | Path,
    *,
    parent_limit: int = 0,
    expected_parent_count: int = DEFAULT_EXPECTED_PARENT_COUNT,
    forbid_calibration_test: bool = True,
) -> tuple[list[TrainParent], list[TrainParent]]:
    path = Path(train_csv).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Strict train CSV is missing or empty: {path}")
    if forbid_calibration_test:
        _reject_non_train_path(path, description="Strict train CSV")
    rows = _read_csv(path)
    if int(expected_parent_count) > 0 and len(rows) != int(expected_parent_count):
        raise ValueError(
            f"Strict train row count mismatch: expected={expected_parent_count}, "
            f"found={len(rows)}."
        )
    if int(parent_limit) < 0:
        raise ValueError("parent_limit must be non-negative.")
    required = {"molecule_id", "label", "split"}
    smiles_col = "smiles" if rows and "smiles" in rows[0] else "parent_smiles"
    required.add(smiles_col)
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Strict train CSV is missing columns: {missing}")
    parents: list[TrainParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        split = row.get("split")
        label = _exact_csv_class_integer(
            row.get("label"),
            path=path,
            row_number=row_number,
            field="label",
        )
        if not parent_id or parent_id in seen:
            raise ValueError(f"Missing/duplicate molecule_id at {path}:{row_number}")
        if type(split) is not str or split != "train":
            raise ValueError(
                f"Strict GlobalGCE input must be train-only, found split={split!r}."
            )
        if label != SOURCE_LABEL:
            raise ValueError(
                f"Strict GlobalGCE source label must be {SOURCE_LABEL}."
            )
        teacher_pred = row.get("teacher_pred")
        if teacher_pred not in (None, "") and _exact_csv_class_integer(
            teacher_pred,
            path=path,
            row_number=row_number,
            field="teacher_pred",
        ) != 1:
            raise ValueError("Strict source parent is not teacher-predicted label 1.")
        if "teacher_correct" in row and not _bool_value(row.get("teacher_correct")):
            raise ValueError("Strict source parent is not teacher-correct.")
        canonical = _canonical_smiles(str(row.get(smiles_col) or ""))
        if canonical is None:
            raise ValueError(f"Invalid parent SMILES at {path}:{row_number}")
        seen.add(parent_id)
        parents.append(TrainParent(parent_id, canonical, label, "train"))
    parents.sort(key=lambda row: row.parent_id)
    selected = parents[: int(parent_limit)] if int(parent_limit) else list(parents)
    return parents, selected


def train_cohort_hash(parents: Sequence[TrainParent]) -> str:
    payload = [
        {
            "parent_id": parent.parent_id,
            "canonical_smiles": parent.smiles,
            "label": parent.label,
            "split": parent.split,
        }
        for parent in parents
    ]
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _stable_split(
    parents: Sequence[TrainParent],
    *,
    seed: int,
    val_fraction: float = 0.20,
) -> tuple[list[int], list[int]]:
    ordered = sorted(
        range(len(parents)),
        key=lambda index: hashlib.sha256(
            f"{seed}\t{parents[index].parent_id}".encode("utf-8")
        ).hexdigest(),
    )
    val_count = max(1, int(round(len(ordered) * float(val_fraction))))
    if val_count >= len(ordered):
        val_count = max(1, len(ordered) - 1)
    val = sorted(ordered[:val_count])
    train = sorted(ordered[val_count:])
    if not train or not val:
        raise ValueError("Native GlobalGCE split requires at least two parents.")
    return train, val


def _ids_hash(parents: Sequence[TrainParent], indices: Sequence[int]) -> str:
    ids = sorted(parents[index].parent_id for index in indices)
    return hashlib.sha256(_json_dumps(ids).encode("utf-8")).hexdigest()


def _ordered_parent_cohort_hash(
    parents: Sequence[TrainParent],
    indices: Sequence[int] | None = None,
) -> str:
    selected = range(len(parents)) if indices is None else indices
    payload = [
        {
            "position": position,
            "parent_id": parents[index].parent_id,
            "canonical_smiles": parents[index].smiles,
            "label": parents[index].label,
            "split": parents[index].split,
        }
        for position, index in enumerate(selected)
    ]
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _cohort_resume_identity(
    parents: Sequence[TrainParent],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> dict[str, Any]:
    ordered_indices = [*train_idx, *val_idx]
    if (
        not train_idx
        or not val_idx
        or len(set(ordered_indices)) != len(ordered_indices)
        or any(
            type(index) is not int or not 0 <= index < len(parents)
            for index in ordered_indices
        )
    ):
        raise ValueError("GlobalGCE resume cohort indices are invalid.")
    return {
        "count": len(ordered_indices),
        "ordered_sha256": _ordered_parent_cohort_hash(
            parents,
            ordered_indices,
        ),
        "train_count": len(train_idx),
        "train_ordered_sha256": _ordered_parent_cohort_hash(parents, train_idx),
        "val_count": len(val_idx),
        "val_ordered_sha256": _ordered_parent_cohort_hash(parents, val_idx),
    }


def _identity_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _official_source_resume_identity(official_src: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in (
        "main.py",
        "models/GTGNN.py",
        "models/GlobalGCE.py",
        "models/models_utils.py",
        "models/fsg.py",
        "data/data_preprocess.py",
        "data/dataset.py",
    ):
        path = official_src / relative
        if path.is_file():
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    if not files:
        raise ValueError("GlobalGCE official source identity is empty.")
    payload: dict[str, Any] = {
        "schema_version": "globalgce_official_source_resume_identity_v1",
        "root": str(official_src),
        "files": files,
    }
    payload["identity_sha256"] = _identity_sha256(payload)
    return payload


def _frozen_gine_resume_identity(
    checkpoint_root: Path,
    *,
    expected_dataset_name: str,
    expected_num_classes: int,
    expected_source_label: int,
) -> dict[str, Any]:
    from src.data.dataset_registry import normalize_dataset_id
    from src.oracles.gnn_oracle import verify_checkpoint_bundle

    root = _descriptor_path_or_resolve(checkpoint_root, strict=True)
    audit = verify_checkpoint_bundle(root, verify_hashes=True)
    manifest_path = root / "sha256sums.txt"
    declared: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or Path(relative).name != relative
            or relative in declared
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Frozen GINE checksum manifest is not canonical.")
        declared[relative] = digest
    inventory = []
    for relative, digest in sorted(declared.items()):
        path = root / relative
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise ValueError("Frozen GINE checksum inventory changed.")
        inventory.append(
            {
                "name": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    temperature_path = root / "temperature_scaling.json"
    temperature_payload = _read_json(temperature_path)
    temperature = temperature_payload.get("temperature")
    if (
        type(temperature) not in (int, float)
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("Frozen GINE temperature identity is invalid.")
    model_card = audit.get("model_card")
    try:
        expected_dataset = normalize_dataset_id(expected_dataset_name)
        observed_dataset = normalize_dataset_id(
            model_card.get("dataset") if type(model_card) is dict else ""
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen GINE model-card dataset identity changed.") from exc
    if (
        type(model_card) is not dict
        or observed_dataset != expected_dataset
        or model_card.get("backbone") != "gine"
        or type(model_card.get("num_classes")) is not int
        or model_card["num_classes"] != expected_num_classes
        or type(model_card.get("source_label")) is not int
        or model_card["source_label"] != expected_source_label
    ):
        raise ValueError("Frozen GINE model-card class identity changed.")
    payload = {
        "schema_version": "globalgce_frozen_gine_resume_identity_v1",
        "backend": "frozen_gine",
        "checkpoint_root": str(root),
        "checkpoint_id": _sha256_file(root / "model.pt"),
        "dataset": expected_dataset,
        "num_classes": expected_num_classes,
        "source_label": expected_source_label,
        "temperature_hex": float(temperature).hex(),
        "temperature_scaling_sha256": _sha256_file(temperature_path),
        "sha256sums_sha256": _sha256_file(manifest_path),
        "inventory": inventory,
        "inventory_sha256": _identity_sha256({"files": inventory}),
    }
    payload["identity_sha256"] = _identity_sha256(payload)
    return payload


def _build_training_resume_identity(
    *,
    dataset_name: str,
    num_classes: int,
    source_label: int,
    target_label: int,
    official_src: Path,
    native_train_csv: Path,
    native_parents: Sequence[TrainParent],
    native_train_idx: Sequence[int],
    native_val_idx: Sequence[int],
    source_parents: Sequence[TrainParent],
    source_train_idx: Sequence[int],
    source_val_idx: Sequence[int],
    frozen_gine_checkpoint: Path | None,
    seed: int,
    epochs: int,
    top_k_native: int,
    learning_rate: float,
    dropout: float,
    min_freq: int | None,
    gspan_flush_every: int,
    gspan_max_in_memory_candidates: int,
    gspan_exact_top_k_pruning: bool,
    gspan_adoption_identity: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if frozen_gine_checkpoint is not None:
        oracle_identity = _frozen_gine_resume_identity(
            frozen_gine_checkpoint,
            expected_dataset_name=dataset_name,
            expected_num_classes=num_classes,
            expected_source_label=source_label,
        )
    else:
        native_csv_identity = {
            "path": str(native_train_csv),
            "bytes": native_train_csv.stat().st_size,
            "sha256": _sha256_file(native_train_csv),
        }
        oracle_identity: dict[str, Any] = {
            "schema_version": "globalgce_native_binary_gtgnn_resume_identity_v1",
            "backend": "official_native_gtgnn",
            "num_classes": 2,
            "native_train_csv": native_csv_identity,
        }
        oracle_identity["identity_sha256"] = _identity_sha256(oracle_identity)
    identity = {
        "schema_version": GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "source_label": source_label,
        "target_label": target_label,
        "oracle_identity": oracle_identity,
        "native_train_cohort": _cohort_resume_identity(
            native_parents,
            train_idx=native_train_idx,
            val_idx=native_val_idx,
        ),
        "source_train_cohort": _cohort_resume_identity(
            source_parents,
            train_idx=source_train_idx,
            val_idx=source_val_idx,
        ),
        "official_source_identity": _official_source_resume_identity(official_src),
        "training_config": {
            "seed": seed,
            "epochs": epochs,
            "top_k_native": top_k_native,
            "learning_rate_hex": float(learning_rate).hex(),
            "dropout_hex": float(dropout).hex(),
            "min_freq": min_freq,
            "gspan_flush_every": gspan_flush_every,
            "gspan_max_in_memory_candidates": gspan_max_in_memory_candidates,
            "gspan_exact_top_k_pruning": gspan_exact_top_k_pruning,
            "gspan_adoption_identity": gspan_adoption_identity,
        },
    }
    return normalize_globalgce_training_resume_identity(identity)


def _load_general_train_rows(
    path: Path,
    *,
    allowed_parent_ids: Sequence[str] | None = None,
    num_classes: int = 2,
    payload: bytes | None = None,
) -> list[TrainParent]:
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("Native train num_classes must be an exact integer >= 2.")
    _reject_non_train_path(path, description="Native GNN train CSV")
    rows = _read_csv(path) if payload is None else _read_csv_payload(payload)
    if not rows:
        raise ValueError(f"Native GNN train CSV is empty: {path}")
    required = {"molecule_id", "label", "split"}
    smiles_col = next(
        (
            column
            for column in (
                "model_smiles",
                "canonical_smiles",
                "smiles",
                "parent_smiles",
            )
            if column in rows[0]
        ),
        "model_smiles",
    )
    required.add(smiles_col)
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Native GNN train CSV is missing columns: {missing}")
    parents: list[TrainParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        split = row.get("split")
        if not parent_id or parent_id in seen:
            raise ValueError(f"Invalid native train molecule_id at {path}:{row_number}")
        if type(split) is not str or split != "train":
            raise ValueError("Native GNN training data contains non-train split.")
        label = _exact_csv_class_integer(
            row.get("label"),
            path=path,
            row_number=row_number,
            field="label",
        )
        if not 0 <= label < num_classes:
            raise ValueError(
                "Native GNN training label is outside its frozen class range."
            )
        canonical = _canonical_smiles(str(row.get(smiles_col) or ""))
        if canonical is None:
            raise ValueError(f"Invalid native train SMILES at {path}:{row_number}")
        seen.add(parent_id)
        parents.append(TrainParent(parent_id, canonical, label, "train"))
    if allowed_parent_ids is not None:
        normalized_ids = tuple(str(value).strip() for value in allowed_parent_ids)
        if not normalized_ids or any(not value for value in normalized_ids):
            raise ValueError("Native GNN parent-ID filter must be non-empty.")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("Native GNN parent-ID filter contains duplicates.")
        requested = set(normalized_ids)
        missing = sorted(requested - seen)
        if missing:
            raise ValueError(
                "Native GNN parent-ID filter is not a subset of the train CSV: "
                f"missing_count={len(missing)}, examples={missing[:5]}"
            )
        parents = [parent for parent in parents if parent.parent_id in requested]
    if {parent.label for parent in parents} != set(range(num_classes)):
        raise ValueError(
            "Official GlobalGCE native metadata requires every frozen class."
        )
    parents.sort(key=lambda row: row.parent_id)
    return parents


def _resolve_official_src(official_root: str | Path) -> Path:
    root = _descriptor_path_or_resolve(official_root)
    candidates = (root, root / "src")
    for candidate in candidates:
        if (
            (candidate / "main.py").is_file()
            and (candidate / "models/GlobalGCE.py").is_file()
            and (candidate / "data/data_preprocess.py").is_file()
        ):
            if "saved_results" in candidate.parts:
                raise ValueError("Official saved_results cannot be used as generation input.")
            return candidate
    raise FileNotFoundError(f"Invalid GlobalGCE official source root: {root}")


def _bond_name(bond: Any) -> str:
    if bond.GetBondType() == Chem.BondType.TRIPLE:
        return "triple"
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "double"
    return "single"


def _kekulized_molecule(smiles: str) -> Any:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot parse molecule for native GlobalGCE: {smiles}")
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except Exception as exc:
        raise ValueError(f"Cannot kekulize molecule for GlobalGCE: {smiles}") from exc
    return molecule


def _source_atom_attribute_sidecar(
    smiles: str,
    native_molecule: Any,
) -> list[dict[str, Any]]:
    """Capture atom properties omitted by the official Mutagenicity features.

    `_build_dense_dataset` writes each RDKit atom to the native tensor slot
    with the same `atom.GetIdx()`. This construction provenance gives a unique
    atom identity even for symmetric molecules where graph isomorphism alone
    would have multiple automorphisms.
    """

    source_molecule = Chem.MolFromSmiles(smiles)
    if source_molecule is None:
        raise GlobalGCEMutagenicityCodecError(
            f"Cannot construct source atom attributes for SMILES: {smiles}"
        )
    if source_molecule.GetNumAtoms() != native_molecule.GetNumAtoms():
        raise GlobalGCEMutagenicityCodecError(
            "Source/native atom count mismatch while constructing atom "
            f"attribute sidecar for {smiles!r}."
        )
    sidecar: list[dict[str, Any]] = []
    for native_index in range(native_molecule.GetNumAtoms()):
        source_atom = source_molecule.GetAtomWithIdx(native_index)
        native_atom = native_molecule.GetAtomWithIdx(native_index)
        if source_atom.GetAtomicNum() != native_atom.GetAtomicNum():
            raise GlobalGCEMutagenicityCodecError(
                "Source/native atomic-number mismatch at preserved atom "
                f"index={native_index} for {smiles!r}."
            )
        sidecar.append(
            {
                "native_node_index": native_index,
                "source_atom_index": source_atom.GetIdx(),
                "atomic_num": source_atom.GetAtomicNum(),
                "formal_charge": source_atom.GetFormalCharge(),
                "is_aromatic": source_atom.GetIsAromatic(),
                "num_explicit_hs": source_atom.GetNumExplicitHs(),
                "isotope": source_atom.GetIsotope(),
                "chiral_tag": int(source_atom.GetChiralTag()),
                "no_implicit": source_atom.GetNoImplicit(),
                "attribute_source": "source_anchored",
            }
        )
    source_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in source_molecule.GetBonds()
    }
    native_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in native_molecule.GetBonds()
    }
    if source_pairs != native_pairs:
        raise GlobalGCEMutagenicityCodecError(
            "Source/native bond topology mismatch under the preserved RDKit "
            f"atom index mapping for {smiles!r}."
        )
    return sidecar


def _load_frozen_gine_feature_schema(
    checkpoint: Path,
    payloads: Mapping[str, bytes] | None,
) -> Any:
    """Load the exact categorical schema carried by one frozen GINE.

    Managed Taste runs provide descriptor-authorized checkpoint bytes, so the
    node vocabulary is derived without reopening the path.  The path fallback
    preserves the existing non-managed frozen-GINE entrypoint.
    """

    from src.data.molecular_graph_featurizer import MolecularFeatureSchema

    try:
        raw = (
            payloads["feature_schema.json"]
            if payloads is not None
            else (checkpoint / "feature_schema.json").read_bytes()
        )
        decoded = json.loads(raw.decode("utf-8"))
        if type(decoded) is not dict:
            raise TypeError("feature schema is not one JSON object")
        return MolecularFeatureSchema.from_dict(decoded)
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GlobalGCEMutagenicityCodecError(
            "Frozen GINE feature schema is unavailable or malformed."
        ) from exc


def _taste_train_observed_atom_symbols(
    molecules: Sequence[Any],
    *,
    feature_schema: Any,
) -> tuple[str, ...]:
    """Return a lossless train-only atom vocabulary accepted by the GINE.

    Every observed element remains its own official-GlobalGCE node category.
    Ordering by atomic number is deterministic and independent of CSV order.
    Unknown-bucket encoding is explicitly forbidden because it would merge
    chemically distinct elements before the frozen classifier sees them.
    """

    atomic_fields = [
        field
        for field in feature_schema.node_fields
        if str(field.name) == "atomic_num"
    ]
    if len(atomic_fields) != 1:
        raise GlobalGCEMutagenicityCodecError(
            "Taste frozen GINE must expose exactly one atomic_num field."
        )
    atomic_field = atomic_fields[0]
    observed: dict[int, str] = {}
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            atomic_number = int(atom.GetAtomicNum())
            symbol = str(atom.GetSymbol())
            prior = observed.setdefault(atomic_number, symbol)
            if prior != symbol:
                raise GlobalGCEMutagenicityCodecError(
                    "RDKit returned inconsistent symbols for one atomic number."
                )
    if not observed:
        raise GlobalGCEMutagenicityCodecError(
            "Taste native train cohort contains no atoms."
        )
    unsupported = [
        {"atomic_number": atomic_number, "symbol": observed[atomic_number]}
        for atomic_number in sorted(observed)
        if str(atomic_number) not in atomic_field.values[: atomic_field.unknown_index]
    ]
    if unsupported:
        raise GlobalGCEMutagenicityCodecError(
            "Taste train atoms are absent from the exact frozen GINE atomic_num "
            f"vocabulary; unknown-bucket mapping is forbidden: {unsupported}"
        )
    return tuple(observed[atomic_number] for atomic_number in sorted(observed))


def _build_dense_dataset(
    parents: Sequence[TrainParent],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    torch_module: Any,
    atom_symbols: Sequence[str] | None = None,
    max_num_nodes: int | None = None,
    dataset_name: str = DATASET_NAME,
    num_classes: int = 2,
    source_label: int = SOURCE_LABEL,
    frozen_gine_feature_schema: Any | None = None,
    atom_vocabulary_source: str | None = None,
    atom_vocabulary_feature_schema_sha256: str | None = None,
) -> _DenseMoleculeDataset:
    if (
        type(num_classes) is not int
        or num_classes < 2
        or type(source_label) is not int
        or not 0 <= source_label < num_classes
    ):
        raise ValueError("Dense GlobalGCE class contract is invalid.")
    molecules = [_kekulized_molecule(parent.smiles) for parent in parents]
    source_atom_attributes = [
        _source_atom_attribute_sidecar(parent.smiles, molecule)
        for parent, molecule in zip(parents, molecules)
    ]
    symbols = list(atom_symbols or ())
    if not symbols:
        present_symbols = {
            atom.GetSymbol()
            for molecule in molecules
            for atom in molecule.GetAtoms()
        }
        if get_dataset_spec(dataset_name).dataset_id == "tastemolnet":
            if frozen_gine_feature_schema is None:
                raise GlobalGCEMutagenicityCodecError(
                    "Taste native GlobalGCE tensors require the exact frozen "
                    "GINE feature schema."
                )
            symbols = list(
                _taste_train_observed_atom_symbols(
                    molecules,
                    feature_schema=frozen_gine_feature_schema,
                )
            )
            atom_vocabulary_source = TASTE_EXACT_ATOM_VOCABULARY_SOURCE
            atom_vocabulary_feature_schema_sha256 = str(
                frozen_gine_feature_schema.to_dict()["schema_sha256"]
            )
        else:
            official_symbols = tuple(
                OFFICIAL_MUTAGENICITY_NODE_LABEL_TO_SYMBOL.values()
            )
            unknown_symbols = present_symbols - set(official_symbols)
            if unknown_symbols:
                raise GlobalGCEMutagenicityCodecError(
                    "Current native train molecules contain atom symbols absent "
                    "from official Mutagenicity node metadata: "
                    f"{sorted(unknown_symbols)}"
                )
            symbols = [
                symbol for symbol in official_symbols if symbol in present_symbols
            ]
            atom_vocabulary_source = (
                "official_mutagenicity_train_observed_exact_symbols_v1"
            )
    symbol_index = {symbol: index + 1 for index, symbol in enumerate(symbols)}
    for molecule in molecules:
        unknown = {
            atom.GetSymbol()
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() not in symbol_index
        }
        if unknown:
            raise ValueError(f"Native generation contains unseen atom symbols: {unknown}")
    maximum = int(max_num_nodes or max(molecule.GetNumAtoms() for molecule in molecules))
    if any(molecule.GetNumAtoms() > maximum for molecule in molecules):
        raise ValueError("Native dataset max_num_nodes is too small.")
    node_dim = len(symbols) + 1
    edge_dim = 4
    edge_slots = maximum * (maximum - 1) // 2
    feat = torch_module.zeros(
        (len(parents), maximum, node_dim),
        dtype=torch_module.float32,
    )
    adj = torch_module.zeros(
        (len(parents), maximum, maximum),
        dtype=torch_module.float32,
    )
    edge_attr = torch_module.zeros(
        (len(parents), edge_slots, edge_dim),
        dtype=torch_module.float32,
    )
    edge_attr[:, :, 0] = 1.0
    labels = torch_module.empty(len(parents), dtype=torch_module.long)
    num_nodes = torch_module.empty(len(parents), dtype=torch_module.long)
    num_edges = torch_module.empty(len(parents), dtype=torch_module.long)
    bond_index = {"single": 1, "double": 2, "triple": 3}
    for graph_index, (parent, molecule) in enumerate(zip(parents, molecules)):
        feat[graph_index, :, 0] = 1.0
        for atom in molecule.GetAtoms():
            atom_index = atom.GetIdx()
            feat[graph_index, atom_index, 0] = 0.0
            feat[graph_index, atom_index, symbol_index[atom.GetSymbol()]] = 1.0
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[graph_index, left, right] = 1.0
            adj[graph_index, right, left] = 1.0
            high, low = max(left, right), min(left, right)
            edge_position = (high - 1) * high // 2 + low
            edge_attr[graph_index, edge_position, 0] = 0.0
            edge_attr[
                graph_index,
                edge_position,
                bond_index[_bond_name(bond)],
            ] = 1.0
        # Official GlobalGCE recourses internal class 0 graphs toward class 1.
        labels[graph_index] = 0 if parent.label == source_label else 1
        num_nodes[graph_index] = molecule.GetNumAtoms()
        num_edges[graph_index] = molecule.GetNumBonds() * 2
    return _DenseMoleculeDataset(
        torch_module=torch_module,
        parent_ids=[parent.parent_id for parent in parents],
        feat=feat,
        adj=adj,
        edge_attr=edge_attr,
        labels=labels,
        num_nodes=num_nodes,
        num_edges=num_edges,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        atom_symbols=symbols,
        bond_names=("no_edge", "single", "double", "triple"),
        source_atom_attributes=source_atom_attributes,
        dataset_name=dataset_name,
        num_classes=num_classes,
        atom_vocabulary_source=(
            atom_vocabulary_source
            or "explicit_exact_atom_symbols_v1"
        ),
        atom_vocabulary_feature_schema_sha256=(
            atom_vocabulary_feature_schema_sha256
        ),
    )


_OFFICIAL_SOURCE_AUTHORITY_KEYS = frozenset(
    {"device", "inode", "bytes", "sha256"}
)


def _normalize_official_source_authority(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    if type(value) is not dict or not value:
        raise ValueError(
            "Official GlobalGCE source authority must be one nonempty dict."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for relative, evidence in value.items():
        path = Path(relative) if type(relative) is str else Path(".")
        if (
            type(relative) is not str
            or not relative
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix != ".py"
            or type(evidence) is not dict
            or set(evidence) != _OFFICIAL_SOURCE_AUTHORITY_KEYS
            or any(
                type(evidence.get(field)) is not int or evidence[field] < minimum
                for field, minimum in (
                    ("device", 0),
                    ("inode", 1),
                    ("bytes", 0),
                )
            )
            or type(evidence.get("sha256")) is not str
            or len(evidence["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in evidence["sha256"]
            )
        ):
            raise ValueError("Official GlobalGCE source authority is malformed.")
        normalized[relative] = dict(evidence)
    return normalized


def _hash_open_regular_source(path: Path) -> dict[str, Any]:
    named = os.stat(path, follow_symlinks=False)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or any(
                int(getattr(named, field)) != int(getattr(observed, field))
                for field in ("st_dev", "st_ino", "st_size")
            )
        ):
            raise RuntimeError("Official GlobalGCE imported source leaf changed.")
        size = int(observed.st_size)
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not block:
                raise RuntimeError("Official GlobalGCE imported source ended early.")
            digest.update(block)
            offset += len(block)
        if os.pread(descriptor, 1, size):
            raise RuntimeError("Official GlobalGCE imported source grew while hashing.")
        return {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "bytes": size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _reject_official_import_artifacts(official_src: Path) -> None:
    """Refuse bytecode/native shadows before Python can select them."""

    for entry in os.scandir(official_src):
        if entry.name == "__pycache__" or Path(entry.name).suffix.lower() in {
            ".pyc",
            ".pyo",
            ".so",
            ".dylib",
        }:
            raise RuntimeError(
                "Official GlobalGCE source root contains runtime code artifacts."
            )
    for relative_root in ("models", "data"):
        root = official_src / relative_root
        if not root.is_dir():
            continue
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            if "__pycache__" in directory_names:
                raise RuntimeError(
                    "Official GlobalGCE source closure contains __pycache__."
                )
            for name in file_names:
                if Path(name).suffix.lower() in {".pyc", ".pyo", ".so", ".dylib"}:
                    raise RuntimeError(
                        "Official GlobalGCE source closure contains runtime code "
                        "artifacts."
                    )


def _is_official_module_name(name: str) -> bool:
    return (
        name == "models"
        or name.startswith("models.")
        or name == "data"
        or name.startswith("data.")
        or name == "utils"
    )


def _official_module_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if _is_official_module_name(name)
        )
    )


def _validate_official_module(
    name: str,
    module: Any,
    *,
    official_src: Path,
    expected_source_authority: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        paths = tuple(str(path) for path in getattr(module, "__path__", ()))
        if (
            name not in {"models", "data"}
            or paths != (str(official_src / name),)
        ):
            raise RuntimeError(
                "Cached official GlobalGCE module has no exact source origin."
            )
        return
    path = Path(str(module_file))
    if path.suffix != ".py":
        raise RuntimeError(
            "Official GlobalGCE module was not loaded from Python source."
        )
    try:
        relative = path.relative_to(official_src).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "Cached official GlobalGCE module originated outside the held checkout."
        ) from exc
    if not (
        relative == "utils.py"
        or relative.startswith("models/")
        or relative.startswith("data/")
    ):
        raise RuntimeError("Official GlobalGCE module source origin changed.")
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    if (
        spec is None
        or type(getattr(spec, "origin", None)) is not str
        or Path(spec.origin) != path
        or not isinstance(loader, importlib.machinery.SourceFileLoader)
        or Path(str(getattr(loader, "path", ""))) != path
    ):
        raise RuntimeError(
            "Official GlobalGCE module loader/source origin changed."
        )
    observed = _hash_open_regular_source(path)
    if expected_source_authority is not None:
        expected = expected_source_authority.get(relative)
        if expected is None or observed != dict(expected):
            raise RuntimeError(
                "Official GlobalGCE module differs from the held source authority."
            )


def _validate_official_module_closure(
    *,
    official_src: Path,
    expected_source_authority: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    names = _official_module_names()
    if not names or "models" not in names:
        raise RuntimeError("Official GlobalGCE model module closure is incomplete.")
    for name in names:
        _validate_official_module(
            name,
            sys.modules[name],
            official_src=official_src,
            expected_source_authority=expected_source_authority,
        )


def _validate_named_official_source_authority(
    official_src: Path,
    authority: Mapping[str, Mapping[str, Any]],
) -> None:
    for relative, expected in authority.items():
        observed = _hash_open_regular_source(official_src / relative)
        if observed != dict(expected):
            raise RuntimeError(
                "Official GlobalGCE named source differs from held authority."
            )


def _import_official_modules(
    official_src: Path,
    *,
    expected_source_authority: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    official_src = _descriptor_path_or_resolve(official_src, strict=True)
    normalized_authority = _normalize_official_source_authority(
        expected_source_authority
    )
    _reject_official_import_artifacts(official_src)
    if normalized_authority is not None:
        _validate_named_official_source_authority(
            official_src,
            normalized_authority,
        )

    # A prior branch may leave a valid official closure in this interpreter.
    # Validate it before removal so a preloaded attacker cannot be silently
    # laundered, then import a fresh branch-local closure from the held root.
    prior_names = _official_module_names()
    if prior_names:
        for name in prior_names:
            _validate_official_module(
                name,
                sys.modules[name],
                official_src=official_src,
                expected_source_authority=normalized_authority,
            )
        if normalized_authority is not None:
            _validate_named_official_source_authority(
                official_src,
                normalized_authority,
            )
        for name in reversed(prior_names):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

    source_text = str(official_src)
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.path[:] = [source_text, *(entry for entry in sys.path if entry != source_text)]
    sys.dont_write_bytecode = True
    try:
        models_utils = importlib.import_module("models.models_utils")
        gtgnn = importlib.import_module("models.GTGNN")
        globalgce = importlib.import_module("models.GlobalGCE")
        fsg_module = importlib.import_module("models.fsg")
        gspan_module = importlib.import_module("models.gSpan.gSpan")
        _validate_official_module_closure(
            official_src=official_src,
            expected_source_authority=normalized_authority,
        )
        if normalized_authority is not None:
            _validate_named_official_source_authority(
                official_src,
                normalized_authority,
            )
        return {
            "GTGNN": gtgnn.GTGNN,
            "GlobalGCE": globalgce.GlobalGCE,
            "generate_cfs": globalgce.generate_cfs,
            "train_globalgce": models_utils.train_globalgce,
            "test_globalgce": models_utils.test_globalgce,
            "fsg_module": fsg_module,
            "gspan_module": gspan_module,
            "gtgnn_module": gtgnn,
            "globalgce_module": globalgce,
            "models_utils_module": models_utils,
        }
    except BaseException:
        for name in reversed(_official_module_names()):
            sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
        sys.path[:] = original_path


def validate_official_globalgce_api_signatures(
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    globalgce = modules.get("globalgce_module")
    models_utils = modules.get("models_utils_module")
    gtgnn = modules.get("gtgnn_module")
    fsg = modules.get("fsg_module")
    gspan = modules.get("gspan_module")
    if any(value is None for value in (globalgce, models_utils, gtgnn, fsg, gspan)):
        raise RuntimeError("BLOCKED_OFFICIAL_API_SIGNATURE_DRIFT")
    entrypoints = {
        "models.GTGNN.GTGNN": gtgnn.GTGNN,
        "models.GlobalGCE.GlobalGCE": globalgce.GlobalGCE,
        "models.GlobalGCE.GlobalGCE.get_fs_expanded_data": (
            globalgce.GlobalGCE.get_fs_expanded_data
        ),
        "models.GlobalGCE.GlobalGCE.get_rules": globalgce.GlobalGCE.get_rules,
        "models.GlobalGCE.GlobalGCE.run_one_batch": (
            globalgce.GlobalGCE.run_one_batch
        ),
        "models.GlobalGCE.generate_cfs": globalgce.generate_cfs,
        "models.GlobalGCE.concate_inputs_with_local_recourse": (
            globalgce.concate_inputs_with_local_recourse
        ),
        "models.fsg.FrequentSubgraphGenerator": fsg.FrequentSubgraphGenerator,
        "models.gSpan.gSpan.gSpan": gspan.gSpan,
        "models.models_utils.train_globalgce": models_utils.train_globalgce,
        "models.models_utils.test_globalgce": models_utils.test_globalgce,
    }
    observed: dict[str, str] = {}
    for name, target in entrypoints.items():
        signature = inspect.signature(target)
        if any(
            parameter.kind
            in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in signature.parameters.values()
        ):
            raise RuntimeError("BLOCKED_OFFICIAL_API_SIGNATURE_DRIFT")
        rendered = str(signature)
        observed[name] = rendered
        if rendered != PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES[name]:
            raise RuntimeError("BLOCKED_OFFICIAL_API_SIGNATURE_DRIFT")
    return {
        "schema_version": OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA,
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "signatures": observed,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _module_provenance_entry(
    *,
    name: str,
    module: Any,
    expected_roots: Sequence[Path],
    package_version: str,
) -> dict[str, Any]:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str or not raw or raw.endswith((".pyc", ".pyo")):
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    path = Path(raw)
    try:
        realpath = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE") from exc
    normalized_roots = tuple(root.resolve(strict=True) for root in expected_roots)
    if not any(_path_is_within(realpath, root) for root in normalized_roots):
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    evidence = _hash_open_regular_source(path)
    return {
        "module": name,
        "module_file": str(path),
        "realpath": str(realpath),
        "sha256": evidence["sha256"],
        "device": evidence["device"],
        "inode": evidence["inode"],
        "bytes": evidence["bytes"],
        "package_version": package_version,
        "expected_roots": [str(root) for root in normalized_roots],
    }


def capture_globalgce_module_provenance(
    *,
    official_src: Path,
    expected_source_authority: Mapping[str, Mapping[str, Any]] | None,
    require_isolated: bool,
) -> dict[str, Any]:
    if type(require_isolated) is not bool:
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    if require_isolated and (
        not bool(sys.flags.isolated)
        or not bool(sys.flags.no_user_site)
        or "" in sys.path
    ):
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    normalized_authority = _normalize_official_source_authority(
        expected_source_authority
    )
    if require_isolated and normalized_authority is None:
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    _validate_official_module_closure(
        official_src=official_src,
        expected_source_authority=normalized_authority,
    )
    if normalized_authority is not None:
        _validate_named_official_source_authority(
            official_src,
            normalized_authority,
        )
    try:
        import torch
        import torch_geometric

        bridge = importlib.import_module(
            "src.baselines.globalgce_frozen_gine_bridge"
        )
        oracle = importlib.import_module("src.oracles.gnn_oracle")
        adapter = sys.modules[__name__]
        torch_version = importlib.metadata.version("torch")
        geometric_version = importlib.metadata.version("torch-geometric")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE") from exc
    project_root = Path(__file__).resolve().parents[2]
    site_roots = tuple(
        Path(value)
        for value in site.getsitepackages()
        if _path_is_within(Path(value).resolve(), Path(sys.prefix).resolve())
    )
    if not site_roots:
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    entries: list[dict[str, Any]] = []
    for name in _official_module_names():
        # ``models``/``data`` may be namespace packages in the pinned
        # checkout.  Their exact ``__path__`` is already enforced by
        # ``_validate_official_module_closure``; provenance rows are reserved
        # for real source leaves that can carry an fd-bound SHA256 identity.
        if getattr(sys.modules[name], "__file__", None) is None:
            continue
        entries.append(
            _module_provenance_entry(
                name=name,
                module=sys.modules[name],
                expected_roots=(official_src,),
                package_version=OFFICIAL_GLOBALGCE_COMMIT,
            )
        )
    for name, module, version in (
        ("torch", torch, torch_version),
        ("torch_geometric", torch_geometric, geometric_version),
        (
            "src.baselines.globalgce_mutagenicity_adapter",
            adapter,
            "project-source",
        ),
        (
            "src.baselines.globalgce_frozen_gine_bridge",
            bridge,
            "project-source",
        ),
        ("src.oracles.gnn_oracle", oracle, "project-source"),
    ):
        entries.append(
            _module_provenance_entry(
                name=name,
                module=module,
                expected_roots=(
                    (project_root,)
                    if name.startswith("src.")
                    else site_roots
                ),
                package_version=version,
            )
        )
    entries.sort(key=lambda row: row["module"])
    observed_names = {row["module"] for row in entries}
    if not {
        "models.GTGNN",
        "models.GlobalGCE",
        "models.models_utils",
        "models.fsg",
        "models.gSpan.gSpan",
        "torch",
        "torch_geometric",
        "src.baselines.globalgce_mutagenicity_adapter",
        "src.baselines.globalgce_frozen_gine_bridge",
        "src.oracles.gnn_oracle",
    }.issubset(observed_names):
        raise RuntimeError("BLOCKED_PYTHON_MODULE_PROVENANCE")
    return {
        "schema_version": OFFICIAL_GLOBALGCE_MODULE_PROVENANCE_SCHEMA,
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "isolated_python": bool(sys.flags.isolated),
        "no_user_site": bool(sys.flags.no_user_site),
        "entries": entries,
    }


def _write_canonical_identity_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = (_json_dumps(value) + "\n").encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError("GlobalGCE startup identity document changed.")
        return expected
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("GlobalGCE startup identity write failed.")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return expected


def _train_native_gnn(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    torch_module: Any,
    epochs: int,
    learning_rate: float,
    checkpoint: Path,
) -> dict[str, Any]:
    optimizer = torch_module.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=0.001,
    )
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, correct = model.run_one_batch(batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_correct += int(correct)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch_module.no_grad():
            for batch in val_loader:
                loss, correct = model.run_one_batch(batch)
                val_loss += float(loss.detach().cpu())
                val_correct += int(correct)
        val_loss /= max(len(val_loader), 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_accuracy": train_correct / max(len(train_loader.dataset), 1),
            "val_loss": val_loss,
            "val_accuracy": val_correct / max(len(val_loader.dataset), 1),
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Native GlobalGCE GNN training produced no checkpoint.")
    model.load_state_dict(best_state)
    torch_module.save(best_state, checkpoint)
    return {
        "epochs": int(epochs),
        "best_val_loss": best_loss,
        "last_metrics": history[-1],
        "checkpoint": str(checkpoint.resolve()),
    }


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _argmax_index(values: Any) -> int:
    row = _to_builtin(values)
    if not isinstance(row, list) or not row:
        return int(row)
    return max(range(len(row)), key=lambda index: float(row[index]))


def _edge_vector_position(left: int, right: int) -> int:
    high, low = max(int(left), int(right)), min(int(left), int(right))
    return (high - 1) * high // 2 + low


def globalgce_tensors_to_graph_record(
    feature: Any,
    adjacency: Any,
    edge_attr: Any,
    *,
    metadata: GlobalGCECodecMetadata,
    num_nodes: int | None = None,
) -> dict[str, Any]:
    """Convert native tensors to discrete labels without doing chemistry repair."""

    feature_rows = _to_builtin(feature)
    adjacency_rows = _to_builtin(adjacency)
    if not isinstance(feature_rows, list) or not isinstance(adjacency_rows, list):
        raise GlobalGCEMutagenicityCodecError(
            "GlobalGCE feature/adjacency tensors are not matrix-like."
        )
    matrix_size = min(len(feature_rows), len(adjacency_rows))
    if matrix_size <= 0:
        raise GlobalGCEMutagenicityCodecError("GlobalGCE graph tensor is empty.")
    node_labels = [_argmax_index(feature_rows[index]) for index in range(matrix_size)]
    raw_edge_mask = [[False] * matrix_size for _ in range(matrix_size)]
    self_loop_count = 0
    asymmetric_pairs: list[list[int]] = []
    for left in range(matrix_size):
        row = adjacency_rows[left]
        if not isinstance(row, list) or len(row) < matrix_size:
            raise GlobalGCEMutagenicityCodecError(
                f"GlobalGCE adjacency row {left} has invalid shape."
            )
        if float(row[left]) > 0.5:
            self_loop_count += 1
        for right in range(left + 1, matrix_size):
            forward = float(row[right]) > 0.5
            reverse = float(adjacency_rows[right][left]) > 0.5
            if forward != reverse:
                asymmetric_pairs.append([left, right])
            # Official inputs and decoder outputs are symmetric. An asymmetric
            # result is retained for audit and rejected by the molecule codec.
            raw_edge_mask[left][right] = forward and reverse
            raw_edge_mask[right][left] = forward and reverse

    requested_num_nodes = (
        int(num_nodes) if num_nodes is not None else None
    )
    if requested_num_nodes is not None and not 0 < requested_num_nodes <= matrix_size:
        raise GlobalGCEMutagenicityCodecError(
            f"Invalid native num_nodes={requested_num_nodes} for size={matrix_size}."
        )
    if requested_num_nodes is None:
        active_nodes = [
            index
            for index, label in enumerate(node_labels)
            if label > 0 and any(raw_edge_mask[index])
        ]
    else:
        active_nodes = [
            index
            for index in range(requested_num_nodes)
            if node_labels[index] > 0
        ]
    active_set = set(active_nodes)
    padding_nodes = [
        index for index in range(matrix_size) if index not in active_set
    ]
    edge_mask = [[0] * matrix_size for _ in range(matrix_size)]
    for left in active_nodes:
        for right in active_nodes:
            if left != right and raw_edge_mask[left][right]:
                edge_mask[left][right] = 1

    edge_labels = [0] * (matrix_size * (matrix_size - 1) // 2)
    if edge_attr is not None:
        edge_rows = _to_builtin(edge_attr)
        if not isinstance(edge_rows, list):
            raise GlobalGCEMutagenicityCodecError(
                "GlobalGCE edge tensor is not vector-like."
            )
        for position in range(min(len(edge_labels), len(edge_rows))):
            edge_labels[position] = _argmax_index(edge_rows[position])

    edge_matrix = [[0] * matrix_size for _ in range(matrix_size)]
    for left in active_nodes:
        for right in active_nodes:
            if left >= right or not edge_mask[left][right]:
                continue
            position = _edge_vector_position(left, right)
            label = int(edge_labels[position]) if position < len(edge_labels) else 0
            edge_matrix[left][right] = label
            edge_matrix[right][left] = label
    node_symbols = [
        (
            metadata.atom_symbols[label - 1]
            if index in active_set and 0 < label <= len(metadata.atom_symbols)
            else None
        )
        for index, label in enumerate(node_labels)
    ]
    return {
        "adjacency": edge_mask,
        "node_labels_internal": node_labels,
        "node_symbols": node_symbols,
        "edge_labels_internal_matrix": edge_matrix,
        "active_node_indices": active_nodes,
        "padding_node_indices": padding_nodes,
        "self_loop_count_ignored": self_loop_count,
        "asymmetric_adjacency_pairs": asymmetric_pairs,
        "codec_metadata": metadata.to_dict(),
    }


def _bond_type_from_metadata(name: str) -> Any:
    normalized = str(name).strip().lower()
    mapping = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }
    if normalized not in mapping:
        raise GlobalGCEMutagenicityCodecError(
            f"Unsupported native bond label metadata: {name!r}"
        )
    return mapping[normalized]


def _add_bond_once(
    molecule: Any,
    left: int,
    right: int,
    bond_type: Any,
    seen_pairs: set[tuple[int, int]],
) -> None:
    pair = tuple(sorted((int(left), int(right))))
    if pair[0] == pair[1]:
        return
    if pair in seen_pairs or molecule.GetBondBetweenAtoms(*pair) is not None:
        raise GlobalGCEMutagenicityCodecError(
            f"Duplicate native bond for atom pair={pair}."
        )
    molecule.AddBond(pair[0], pair[1], bond_type)
    seen_pairs.add(pair)


def _source_attributes_by_native_index(
    source_atom_attributes: Sequence[dict[str, Any]] | None,
) -> dict[int, dict[str, Any]]:
    if source_atom_attributes is None:
        raise GlobalGCEMutagenicityCodecError(
            "Source atom attributes are unavailable."
        )
    by_native_index: dict[int, dict[str, Any]] = {}
    source_indices: set[int] = set()
    for raw in source_atom_attributes:
        row = dict(raw)
        try:
            native_index = int(row["native_node_index"])
            source_index = int(row["source_atom_index"])
            atomic_num = int(row["atomic_num"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GlobalGCEMutagenicityCodecError(
                f"Malformed source atom attribute row: {row!r}."
            ) from exc
        if native_index < 0 or source_index < 0 or atomic_num <= 0:
            raise GlobalGCEMutagenicityCodecError(
                f"Invalid source atom attribute row: {row!r}."
            )
        if native_index in by_native_index or source_index in source_indices:
            raise GlobalGCEMutagenicityCodecError(
                "Source atom mapping is not unique: "
                f"native_node_index={native_index}, "
                f"source_atom_index={source_index}."
            )
        by_native_index[native_index] = row
        source_indices.add(source_index)
    return by_native_index


def _apply_source_atom_attributes(atom: Any, attributes: dict[str, Any]) -> None:
    atom.SetFormalCharge(int(attributes.get("formal_charge") or 0))
    atom.SetIsotope(int(attributes.get("isotope") or 0))
    atom.SetChiralTag(
        Chem.rdchem.ChiralType(int(attributes.get("chiral_tag") or 0))
    )
    atom.SetNumExplicitHs(int(attributes.get("num_explicit_hs") or 0))
    atom.SetNoImplicit(bool(attributes.get("no_implicit", False)))


def decode_globalgce_molecule(
    feature: Any,
    adjacency: Any,
    edge_attr: Any,
    *,
    metadata: GlobalGCECodecMetadata,
    num_nodes: int | None = None,
    graph_role: str,
    source_atom_attributes: Sequence[dict[str, Any]] | None,
) -> GlobalGCEGraphDecodeResult:
    """Decode one native graph using its explicit node/edge metadata."""

    if graph_role not in {"source", "generated"}:
        raise ValueError(f"Unsupported GlobalGCE graph role: {graph_role!r}")
    attribute_error_type = (
        "source_codec_attribute_mapping_failed"
        if graph_role == "source"
        else "generated_attribute_ambiguous"
    )
    graph_error_type = (
        "source_codec_structure_mismatch"
        if graph_role == "source"
        else "generated_sanitize_failure"
    )
    try:
        record = globalgce_tensors_to_graph_record(
            feature,
            adjacency,
            edge_attr,
            metadata=metadata,
            num_nodes=num_nodes,
        )
    except Exception as exc:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            graph_error_type,
            str(exc),
            0,
            0,
            {},
        )
    if record["asymmetric_adjacency_pairs"]:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            graph_error_type,
            f"pairs={record['asymmetric_adjacency_pairs'][:5]}",
            0,
            0,
            record,
        )
    active_nodes = list(record["active_node_indices"])
    if not active_nodes:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            graph_error_type,
            "No real nodes.",
            0,
            0,
            record,
        )
    try:
        source_by_native = _source_attributes_by_native_index(
            source_atom_attributes
        )
    except GlobalGCEMutagenicityCodecError as exc:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            attribute_error_type,
            str(exc),
            0,
            0,
            record,
        )
    atom_attribute_audit: list[dict[str, Any]] = []
    for old_index in active_nodes:
        internal_label = int(record["node_labels_internal"][old_index])
        if not 0 < internal_label <= len(metadata.atom_symbols):
            return GlobalGCEGraphDecodeResult(
                False,
                None,
                None,
                False,
                attribute_error_type,
                f"Unknown internal atom label={internal_label}.",
                0,
                0,
                record,
            )
        symbol = metadata.atom_symbols[internal_label - 1]
        generated_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
        source_attributes = source_by_native.get(old_index)
        inherited = (
            source_attributes is not None
            and int(source_attributes["atomic_num"]) == generated_atomic_num
        )
        audit_row = {
            "native_node_index": old_index,
            "source_atom_index": (
                int(source_attributes["source_atom_index"])
                if source_attributes is not None
                else None
            ),
            "atomic_num": generated_atomic_num,
            "formal_charge": (
                int(source_attributes.get("formal_charge") or 0)
                if inherited
                else None
            ),
            "native_atom_label": internal_label,
            "attribute_source": (
                metadata.atom_attribute_source if inherited else None
            ),
            "attributes_inherited": inherited,
        }
        if not inherited:
            audit_row["ambiguity_reason"] = (
                "new_native_node_without_source_identity"
                if source_attributes is None
                else "generated_atom_type_differs_from_source"
            )
            atom_attribute_audit.append(audit_row)
            record["atom_attribute_audit"] = atom_attribute_audit
            return GlobalGCEGraphDecodeResult(
                False,
                None,
                None,
                False,
                attribute_error_type,
                (
                    "Formal charge is not encoded by native Mutagenicity "
                    "features and cannot be inherited for "
                    f"native_node_index={old_index}: {audit_row!r}."
                ),
                0,
                0,
                record,
            )
        atom_attribute_audit.append(audit_row)
    record["atom_attribute_audit"] = atom_attribute_audit
    record["atom_attribute_source"] = metadata.atom_attribute_source
    record["formal_charge_encoded_by_native"] = (
        metadata.formal_charge_encoded_by_native
    )
    record["source_atom_mapping_method"] = (
        metadata.source_atom_mapping_method
    )
    record["source_atom_mapping_unique"] = True
    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    try:
        for old_index in active_nodes:
            internal_label = int(record["node_labels_internal"][old_index])
            if not 0 < internal_label <= len(metadata.atom_symbols):
                raise GlobalGCEMutagenicityCodecError(
                    f"Unknown internal atom label={internal_label}."
                )
            symbol = metadata.atom_symbols[internal_label - 1]
            atom = Chem.Atom(symbol)
            _apply_source_atom_attributes(
                atom,
                source_by_native[old_index],
            )
            old_to_new[old_index] = editable.AddAtom(atom)
        seen_pairs: set[tuple[int, int]] = set()
        for left_position, left in enumerate(active_nodes):
            for right in active_nodes[left_position + 1 :]:
                if not record["adjacency"][left][right]:
                    continue
                edge_label = int(
                    record["edge_labels_internal_matrix"][left][right]
                )
                if edge_label == 0:
                    raise GlobalGCEMutagenicityCodecError(
                        "Adjacency contains an edge whose native edge class is "
                        "the explicit no-edge class."
                    )
                if not 0 < edge_label < len(metadata.bond_names):
                    raise GlobalGCEMutagenicityCodecError(
                        f"Unknown internal edge label={edge_label}."
                    )
                bond_type = _bond_type_from_metadata(
                    metadata.bond_names[edge_label]
                )
                _add_bond_once(
                    editable,
                    old_to_new[left],
                    old_to_new[right],
                    bond_type,
                    seen_pairs,
                )
                if bond_type == Chem.BondType.AROMATIC:
                    editable.GetAtomWithIdx(old_to_new[left]).SetIsAromatic(True)
                    editable.GetAtomWithIdx(old_to_new[right]).SetIsAromatic(True)
        molecule = editable.GetMol()
    except Exception as exc:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            graph_error_type,
            str(exc),
            editable.GetNumAtoms(),
            editable.GetNumBonds(),
            record,
        )
    try:
        Chem.SanitizeMol(molecule)
        smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        error_message = str(exc)
        if graph_role == "source":
            error_type = (
                "source_codec_invalid_valence"
                if "valence" in error_message.lower()
                else "source_codec_structure_mismatch"
            )
        else:
            error_type = (
                "generated_invalid_valence"
                if "valence" in error_message.lower()
                else "generated_sanitize_failure"
            )
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            molecule,
            True,
            error_type,
            error_message,
            molecule.GetNumAtoms(),
            molecule.GetNumBonds(),
            record,
        )
    return GlobalGCEGraphDecodeResult(
        True,
        smiles,
        molecule,
        True,
        None,
        None,
        molecule.GetNumAtoms(),
        molecule.GetNumBonds(),
        record,
    )


def _molecules_structure_match(left: Any, right: Any) -> bool:
    try:
        left_smiles = Chem.MolToSmiles(
            left,
            canonical=True,
            isomericSmiles=False,
        )
        right_smiles = Chem.MolToSmiles(
            right,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return False
    return left_smiles == right_smiles


def probe_source_graph_codec(
    dataset: Any,
    parents: Sequence[TrainParent],
    *,
    atom_attribute_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Round-trip every selected source graph through the production codec."""

    if len(dataset) != len(parents):
        raise GlobalGCEMutagenicityCodecError(
            "Source codec dataset/parent length mismatch: "
            f"dataset={len(dataset)}, parents={len(parents)}."
        )
    metadata = GlobalGCECodecMetadata.from_dataset(dataset)
    valid_count = 0
    structure_count = 0
    invalid_valence_count = 0
    attribute_mapping_failed_count = 0
    charged_atom_count = 0
    charge_audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, parent in enumerate(parents):
        graph = dataset[index]
        result = decode_globalgce_molecule(
            graph["feature"],
            graph["adj"],
            graph.get("edge_attr"),
            metadata=metadata,
            num_nodes=int(_to_builtin(graph["num_nodes"])),
            graph_role="source",
            source_atom_attributes=dataset.source_atom_attributes[index],
        )
        structure_match = False
        parent_molecule = Chem.MolFromSmiles(parent.smiles)
        if result.ok and result.mol is not None and parent_molecule is not None:
            valid_count += 1
            structure_match = (
                result.smiles == parent.smiles
                or _molecules_structure_match(result.mol, parent_molecule)
            )
            if structure_match:
                structure_count += 1
        if result.error_type == "source_codec_invalid_valence":
            invalid_valence_count += 1
        if result.error_type == "source_codec_attribute_mapping_failed":
            attribute_mapping_failed_count += 1
        for atom_audit in result.graph_record.get("atom_attribute_audit", []):
            formal_charge = atom_audit.get("formal_charge")
            if formal_charge is None or int(formal_charge) == 0:
                continue
            charged_atom_count += 1
            if len(charge_audit_rows) < 10:
                charge_audit_rows.append(
                    {
                        "parent_id": parent.parent_id,
                        "native_node_index": atom_audit["native_node_index"],
                        "source_atom_index": atom_audit["source_atom_index"],
                        "atomic_num": atom_audit["atomic_num"],
                        "formal_charge": int(formal_charge),
                        "native_atom_label": atom_audit["native_atom_label"],
                        "attribute_source": atom_audit["attribute_source"],
                    }
                )
        if not result.ok or not structure_match:
            error_type = result.error_type
            if result.ok and not structure_match:
                error_type = "source_codec_structure_mismatch"
            failures.append(
                {
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "decoded_smiles": result.smiles,
                    "codec_ok": result.ok,
                    "structure_match": structure_match,
                    "error_type": error_type,
                    "error_message": result.error_message,
                }
            )
    checked = len(parents)
    summary = {
        "source_codec_checked_rows": checked,
        "source_codec_rdkit_valid_rows": valid_count,
        "source_codec_structure_match_rows": structure_count,
        "source_codec_invalid_valence_rows": invalid_valence_count,
        "source_codec_attribute_mapping_failed_rows": (
            attribute_mapping_failed_count
        ),
        "source_codec_passed": (
            checked > 0
            and valid_count == checked
            and structure_count == checked
        ),
        "source_codec_failure_examples": failures[:10],
        "atom_attribute_source": metadata.atom_attribute_source,
        "formal_charge_encoded_by_native": (
            metadata.formal_charge_encoded_by_native
        ),
        "source_atom_mapping_method": metadata.source_atom_mapping_method,
        "source_atom_mapping_unique": attribute_mapping_failed_count == 0,
        "source_formal_charge_nonzero_atom_count": charged_atom_count,
        "source_atom_attribute_audit_rows": len(charge_audit_rows),
        "codec_metadata": metadata.to_dict(),
    }
    if atom_attribute_audit_path is not None:
        _write_jsonl(
            _descriptor_path_or_resolve(atom_attribute_audit_path),
            charge_audit_rows,
        )
    return summary


def require_source_codec_gate(summary: dict[str, Any]) -> None:
    if summary.get("source_codec_passed") is not True:
        raise GlobalGCEMutagenicityCodecError(
            "GlobalGCE Mutagenicity source codec round-trip failed before "
            f"training: {list(summary.get('source_codec_failure_examples') or [])[:3]}"
        )


def attach_globalgce_generation_dataset(
    augmented_dataset: Any,
    native_dataset: Any,
) -> Any:
    """Retain the real dataset expected by official ``generate_cfs``.

    The official ``AugmentedDataset`` constructor receives ``original_dataset``
    but does not retain it. Official generation later assumes it is available
    as ``dataloader.dataset.dataset``. Attaching that exact object preserves
    the augmented dataset's length, indexing, and item semantics.
    """

    if augmented_dataset is native_dataset:
        raise ValueError(
            "GlobalGCE generation dataset cannot reference itself as its "
            "underlying native dataset."
        )
    _validated_max_num_nodes(native_dataset, role="underlying native dataset")
    augmented_dataset.dataset = native_dataset
    return augmented_dataset


def _validated_max_num_nodes(dataset: Any, *, role: str) -> int:
    if not hasattr(dataset, "max_num_nodes"):
        raise ValueError(
            f"GlobalGCE {role} type={type(dataset).__name__} is missing "
            "required attribute max_num_nodes."
        )
    raw_value = getattr(dataset, "max_num_nodes")
    if isinstance(raw_value, bool):
        raise ValueError(
            f"GlobalGCE {role} has invalid max_num_nodes={raw_value!r}; "
            "expected a finite positive integer."
        )
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GlobalGCE {role} has invalid max_num_nodes={raw_value!r}; "
            "expected a finite positive integer."
        ) from exc
    if (
        not math.isfinite(numeric_value)
        or numeric_value <= 0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"GlobalGCE {role} has invalid max_num_nodes={raw_value!r}; "
            "expected a finite positive integer."
        )
    return int(numeric_value)


def validate_globalgce_generation_loader(dataloader: Any) -> int:
    """Validate the nested dataset contract used by official generation."""

    if not hasattr(dataloader, "dataset"):
        raise ValueError(
            "GlobalGCE generation loader "
            f"type={type(dataloader).__name__} is missing attribute dataset."
        )
    augmented_dataset = dataloader.dataset
    if not hasattr(augmented_dataset, "dataset"):
        raise ValueError(
            "GlobalGCE generation loader dataset "
            f"type={type(augmented_dataset).__name__} is missing underlying "
            "attribute dataset required by official generate_cfs."
        )
    native_dataset = augmented_dataset.dataset
    if native_dataset is augmented_dataset:
        raise ValueError(
            "GlobalGCE generation loader has an invalid self-referential "
            f"dataset type={type(augmented_dataset).__name__}."
        )
    return _validated_max_num_nodes(
        native_dataset,
        role=(
            "generation loader underlying dataset "
            f"type={type(native_dataset).__name__}"
        ),
    )


class _AugmentedGenerationChunk:
    """Index-only view over one parent chunk of the augmented dataset."""

    def __init__(
        self,
        augmented_dataset: Any,
        native_dataset: Any,
        row_indices: Sequence[int],
    ) -> None:
        self.augmented_dataset = augmented_dataset
        self.dataset = native_dataset
        self.row_indices = tuple(int(index) for index in row_indices)
        _validated_max_num_nodes(
            native_dataset,
            role="chunk generation underlying native dataset",
        )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> Any:
        return self.augmented_dataset[self.row_indices[int(index)]]


def _detach_tensor_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _detach_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tensor_tree(item) for item in value)
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach()
    return value


def _augmented_rows_by_source_parent(
    augmented_dataset: Any,
    source_expansion_order: Sequence[int],
    parent_count: int,
) -> list[list[int]]:
    rows_by_parent = [[] for _ in range(int(parent_count))]
    graph_indices = augmented_dataset.graph_idx_list
    for augmented_index in range(len(augmented_dataset)):
        source_position = int(graph_indices[augmented_index])
        if (
            source_position < 0
            or source_position >= len(source_expansion_order)
        ):
            raise RuntimeError(
                "Official GlobalGCE augmented dataset contains invalid source "
                f"position {source_position}."
            )
        source_index = int(source_expansion_order[source_position])
        if source_index < 0 or source_index >= int(parent_count):
            raise RuntimeError(
                "Official GlobalGCE augmented dataset maps outside the selected "
                f"parent cohort: source_index={source_index}."
            )
        rows_by_parent[source_index].append(augmented_index)
    return rows_by_parent


def _prepare_native_and_source_datasets(
    *,
    native_train_csv: Path,
    parents: Sequence[TrainParent],
    seed: int,
    torch_module: Any,
    dataset_name: str = DATASET_NAME,
    native_train_parent_ids: Sequence[str] | None = None,
    native_train_payload: bytes | None = None,
    num_classes: int = 2,
    source_label: int = SOURCE_LABEL,
    frozen_gine_feature_schema: Any | None = None,
) -> tuple[
    list[TrainParent],
    list[int],
    list[int],
    _DenseMoleculeDataset,
    list[int],
    list[int],
    _DenseMoleculeDataset,
]:
    native_parents = _load_general_train_rows(
        native_train_csv,
        allowed_parent_ids=native_train_parent_ids,
        num_classes=num_classes,
        payload=native_train_payload,
    )
    native_train_idx, native_val_idx = _stratified_native_split(
        native_parents,
        seed=int(seed),
    )
    native_dataset = _build_dense_dataset(
        native_parents,
        train_idx=native_train_idx,
        val_idx=native_val_idx,
        test_idx=[],
        torch_module=torch_module,
        dataset_name=dataset_name,
        num_classes=num_classes,
        source_label=source_label,
        frozen_gine_feature_schema=frozen_gine_feature_schema,
    )
    source_train_idx, source_val_idx = _stable_split(parents, seed=int(seed))
    source_dataset = _build_dense_dataset(
        parents,
        train_idx=source_train_idx,
        val_idx=source_val_idx,
        test_idx=[],
        torch_module=torch_module,
        atom_symbols=native_dataset.atom_symbols,
        max_num_nodes=max(
            native_dataset.max_num_nodes,
            max(Chem.MolFromSmiles(parent.smiles).GetNumAtoms() for parent in parents),
        ),
        dataset_name=dataset_name,
        num_classes=num_classes,
        source_label=source_label,
        atom_vocabulary_source=native_dataset.atom_vocabulary_source,
        atom_vocabulary_feature_schema_sha256=(
            native_dataset.atom_vocabulary_feature_schema_sha256
        ),
    )
    return (
        native_parents,
        native_train_idx,
        native_val_idx,
        native_dataset,
        source_train_idx,
        source_val_idx,
        source_dataset,
    )


class OfficialGlobalGCEMutagenicityGenerator:
    """Execute official GlobalGCE components on current train-only tensors."""

    def __init__(
        self,
        official_root: str | Path,
        *,
        native_train_csv: str | Path | None = None,
        dataset_name: str = DATASET_NAME,
        min_freq: int | None = None,
        frozen_gine_checkpoint: str | Path | None = None,
        source_label: int = SOURCE_LABEL,
        target_label: int = TARGET_LABEL,
        num_classes: int = 2,
        native_train_parent_ids: Sequence[str] | None = None,
        frozen_gine_payloads: Mapping[str, bytes] | None = None,
        native_train_payload: bytes | None = None,
        official_source_authority: (
            Mapping[str, Mapping[str, Any]] | None
        ) = None,
        require_isolated_imports: bool = False,
    ) -> None:
        self.official_src = _resolve_official_src(official_root)
        repo_root = Path(__file__).resolve().parents[2]
        configured = native_train_csv or os.environ.get(
            "GLOBALGCE_MUTAGENICITY_NATIVE_TRAIN_CSV"
        )
        self.native_train_csv = _descriptor_path_or_resolve(
            configured or repo_root / DEFAULT_NATIVE_TRAIN_CSV
        )
        dataset_spec = get_dataset_spec(str(dataset_name))
        self.dataset_id = dataset_spec.dataset_id
        self.dataset_name = dataset_spec.display_name
        self.min_freq = int(min_freq) if min_freq is not None else None
        self.frozen_gine_checkpoint = (
            None
            if frozen_gine_checkpoint is None
            else _descriptor_path_or_resolve(frozen_gine_checkpoint)
        )
        if frozen_gine_payloads is not None and (
            type(frozen_gine_payloads) is not dict
            or set(frozen_gine_payloads) != FROZEN_GINE_IN_MEMORY_FILES
            or any(
                type(name) is not str or type(payload) is not bytes or not payload
                for name, payload in frozen_gine_payloads.items()
            )
        ):
            raise ValueError(
                "Descriptor-authorized frozen GINE payloads require the exact "
                "seven-file set of nonempty native bytes."
            )
        self.frozen_gine_payloads = (
            None if frozen_gine_payloads is None else dict(frozen_gine_payloads)
        )
        self.native_train_payload = (
            None if native_train_payload is None else bytes(native_train_payload)
        )
        self.official_source_authority = _normalize_official_source_authority(
            official_source_authority
        )
        if type(require_isolated_imports) is not bool:
            raise ValueError("GlobalGCE isolated-import requirement must be boolean.")
        self.require_isolated_imports = require_isolated_imports
        if native_train_payload is not None and (
            type(native_train_payload) is not bytes or not native_train_payload
        ):
            raise ValueError(
                "Descriptor-authorized native train payload must be nonempty bytes."
            )
        if (
            self.frozen_gine_payloads is not None
            and self.frozen_gine_checkpoint is None
        ):
            raise ValueError(
                "Descriptor-authorized frozen GINE payloads require the exact "
                "seven-file set and one full checkpoint authority."
            )
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("GlobalGCE num_classes must be an exact integer >= 2.")
        if type(source_label) is not int or type(target_label) is not int:
            raise ValueError("GlobalGCE source/target labels must be exact integers.")
        self.num_classes = num_classes
        self.source_label = source_label
        self.target_label = target_label
        self.native_train_parent_ids = (
            None
            if native_train_parent_ids is None
            else tuple(
                sorted(str(value).strip() for value in native_train_parent_ids)
            )
        )
        if self.native_train_parent_ids is not None:
            if not self.native_train_parent_ids or any(
                not value for value in self.native_train_parent_ids
            ):
                raise ValueError("Native GNN parent-ID filter must be non-empty.")
            if len(self.native_train_parent_ids) != len(
                set(self.native_train_parent_ids)
            ):
                raise ValueError("Native GNN parent-ID filter contains duplicates.")
        if (
            not 0 <= self.source_label < self.num_classes
            or not 0 <= self.target_label < self.num_classes
            or self.source_label == self.target_label
        ):
            raise ValueError(
                "GlobalGCE source/target labels must be distinct frozen classes."
            )
        if self.num_classes > 2 and self.frozen_gine_checkpoint is None:
            raise ValueError(
                "Multiclass GlobalGCE requires one frozen GINE; the official "
                "native GTGNN path is binary-only."
            )
        if self.dataset_id == "tastemolnet" and (
            self.num_classes != dataset_spec.num_classes
            or self.source_label != dataset_spec.source_label
            or self.target_label not in {0, 2}
            or self.frozen_gine_checkpoint is None
        ):
            raise ValueError(
                "TasteMolNet GlobalGCE requires the frozen three-class GINE, "
                "Sweet source label 1, and destination 0 or 2."
            )
        if self.min_freq is not None and self.min_freq < 2:
            raise ValueError("GlobalGCE min_freq must be at least two.")

    def config_identity(self) -> dict[str, Any]:
        source_files = {}
        for relative in (
            "main.py",
            "models/GTGNN.py",
            "models/GlobalGCE.py",
            "models/models_utils.py",
            "models/fsg.py",
            "data/data_preprocess.py",
            "data/dataset.py",
        ):
            path = self.official_src / relative
            if path.is_file():
                source_files[relative] = _file_identity(path)
        return {
            "generator_class": type(self).__name__,
            "dataset_name": self.dataset_name,
            "min_freq": self.min_freq,
            "native_train_csv": (
                _file_identity(self.native_train_csv)
                if self.native_train_payload is None
                else {
                    "path": str(self.native_train_csv),
                    "kind": "descriptor_authorized_file_payload",
                    "size": len(self.native_train_payload),
                    "sha256": hashlib.sha256(
                        self.native_train_payload
                    ).hexdigest(),
                }
            ),
            "native_train_parent_id_count": (
                None
                if self.native_train_parent_ids is None
                else len(self.native_train_parent_ids)
            ),
            "native_train_parent_ids_hash": (
                None
                if self.native_train_parent_ids is None
                else hashlib.sha256(
                    _json_dumps(list(self.native_train_parent_ids)).encode("utf-8")
                ).hexdigest()
            ),
            "official_src": str(self.official_src),
            "official_source_files": source_files,
            "official_source_authority_sha256": (
                None
                if self.official_source_authority is None
                else hashlib.sha256(
                    _json_dumps(self.official_source_authority).encode("utf-8")
                ).hexdigest()
            ),
            "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
            "require_isolated_imports": self.require_isolated_imports,
            "prediction_backend": (
                "frozen_gine_differentiable_bridge"
                if self.frozen_gine_checkpoint is not None
                else "official_native_gtgnn"
            ),
            "frozen_gine_checkpoint": (
                _file_identity(self.frozen_gine_checkpoint / "model.pt")
                if self.frozen_gine_checkpoint is not None
                else None
            ),
            "frozen_gine_payload_sha256": (
                None
                if self.frozen_gine_payloads is None
                else {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in sorted(self.frozen_gine_payloads.items())
                }
            ),
            "source_label": self.source_label,
            "target_label": self.target_label,
            "num_classes": self.num_classes,
        }

    def _frozen_feature_schema_for_native_codec(self) -> Any | None:
        if self.frozen_gine_checkpoint is None:
            return None
        return _load_frozen_gine_feature_schema(
            self.frozen_gine_checkpoint,
            self.frozen_gine_payloads,
        )

    def probe_codec(
        self,
        parents: Sequence[TrainParent],
        *,
        seed: int,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.native_train_payload is None and not self.native_train_csv.is_file():
            raise FileNotFoundError(
                "Native GNN train CSV is required for codec metadata: "
                f"{self.native_train_csv}"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("GlobalGCE codec probe requires torch.") from exc
        (
            native_parents,
            native_train_idx,
            native_val_idx,
            native_dataset,
            _source_train_idx,
            _source_val_idx,
            source_dataset,
        ) = _prepare_native_and_source_datasets(
            native_train_csv=self.native_train_csv,
            parents=parents,
            seed=int(seed),
            torch_module=torch,
            dataset_name=self.dataset_name,
            native_train_parent_ids=self.native_train_parent_ids,
            native_train_payload=self.native_train_payload,
            num_classes=self.num_classes,
            source_label=self.source_label,
            frozen_gine_feature_schema=(
                self._frozen_feature_schema_for_native_codec()
            ),
        )
        attribute_audit_path = (
            _descriptor_path_or_resolve(output_path).parent
            / "source_atom_attribute_audit.jsonl"
            if output_path is not None
            else None
        )
        summary = probe_source_graph_codec(
            source_dataset,
            parents,
            atom_attribute_audit_path=attribute_audit_path,
        )
        summary.update(
            {
                "native_train_rows": len(native_parents),
                "native_internal_train_ids_hash": _ids_hash(
                    native_parents,
                    native_train_idx,
                ),
                "native_internal_val_ids_hash": _ids_hash(
                    native_parents,
                    native_val_idx,
                ),
                "native_max_num_nodes": native_dataset.max_num_nodes,
                "calibration_loaded": False,
                "test_loaded": False,
            }
        )
        if output_path is not None:
            _write_json(_descriptor_path_or_resolve(output_path), summary)
        require_source_codec_gate(summary)
        return summary

    def generate(
        self,
        parents: Sequence[TrainParent],
        *,
        output_dir: Path,
        seed: int,
        epochs: int,
        top_k_native: int,
        learning_rate: float,
        dropout: float,
        device: str,
        resume: bool,
        generation_chunk_size: int = 32,
        generation_num_workers: int = 0,
        memory_log_every_chunks: int = 1,
        gspan_flush_every: int = 256,
        gspan_max_in_memory_candidates: int = 256,
        gspan_exact_top_k_pruning: bool = False,
        gspan_adoption_proof: str | Path | None = None,
        start_parent_offset: int = 0,
        on_training_ready: Callable[[dict[str, Any]], None] | None = None,
        on_chunk: (
            Callable[[int, int, int, list[dict[str, Any]]], None] | None
        ) = None,
        rules_only: bool = False,
        expected_resume_checkpoint: Mapping[str, Any] | None = None,
        on_resume_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        after_epoch_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        on_generation_complete: Callable[[], None] | None = None,
    ) -> NativeGenerationResult:
        if expected_resume_checkpoint is not None:
            _normalize_checkpoint_file_evidence(expected_resume_checkpoint)
        for callback_name, callback in (
            ("on_resume_checkpoint", on_resume_checkpoint),
            ("after_epoch_checkpoint", after_epoch_checkpoint),
            ("on_generation_complete", on_generation_complete),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"GlobalGCE {callback_name} must be callable or None.")
        generation_complete_notified = False

        def _notify_generation_complete() -> None:
            nonlocal generation_complete_notified
            if generation_complete_notified:
                raise RuntimeError(
                    "GlobalGCE generation completion callback was repeated."
                )
            generation_complete_notified = True
            if on_generation_complete is not None:
                on_generation_complete()
        if not parents:
            raise ValueError("GlobalGCE requires a non-empty source train cohort.")
        if len({parent.parent_id for parent in parents}) != len(parents):
            raise ValueError("GlobalGCE source train parent IDs must be unique.")
        if any(
            type(parent.label) is not int
            or parent.label != self.source_label
            or parent.split != "train"
            for parent in parents
        ):
            raise ValueError(
                "GlobalGCE source parents must be exact source-label train rows."
            )
        if int(generation_chunk_size) <= 0:
            raise ValueError("generation_chunk_size must be positive.")
        if int(generation_num_workers) < 0:
            raise ValueError("generation_num_workers must be non-negative.")
        if int(memory_log_every_chunks) <= 0:
            raise ValueError("memory_log_every_chunks must be positive.")
        if int(gspan_flush_every) <= 0 or int(gspan_max_in_memory_candidates) <= 0:
            raise ValueError("gSpan streaming limits must be positive.")
        if gspan_adoption_proof is not None and gspan_exact_top_k_pruning:
            raise ValueError(
                "GlobalGCE adoption proof and fresh exact-top-k mining are mutually exclusive."
            )
        if int(start_parent_offset) < 0 or int(start_parent_offset) > len(
            parents
        ):
            raise ValueError(
                "start_parent_offset must be within the selected parent cohort."
            )
        if self.native_train_payload is None and not self.native_train_csv.is_file():
            raise FileNotFoundError(
                "Official GlobalGCE requires the reviewed processed train "
                f"CSV for its native GNN: {self.native_train_csv}"
            )
        modules = _import_official_modules(
            self.official_src,
            expected_source_authority=self.official_source_authority,
        )
        official_api_signature = validate_official_globalgce_api_signatures(
            modules
        )
        python_module_provenance = capture_globalgce_module_provenance(
            official_src=self.official_src,
            expected_source_authority=self.official_source_authority,
            require_isolated=self.require_isolated_imports,
        )
        if self.min_freq is not None:
            # Official fsg.py indexes this process-local mapping directly.
            # The project resolves the value before training and records it in
            # provenance; no upstream source checkout is modified.
            modules["fsg_module"].MIN_FREQ[self.dataset_name] = self.min_freq
        try:
            import numpy as np
            import torch
            from torch.utils.data import DataLoader, Subset
        except ImportError as exc:  # pragma: no cover - HPC dependency
            raise RuntimeError(
                "Official GlobalGCE requires torch, numpy, and torch_geometric."
            ) from exc

        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        resolved_device = torch.device(device)
        frozen_gine_feature_schema = (
            self._frozen_feature_schema_for_native_codec()
        )
        (
            native_parents,
            native_train_idx,
            native_val_idx,
            native_dataset,
            source_train_idx,
            source_val_idx,
            source_dataset,
        ) = _prepare_native_and_source_datasets(
            native_train_csv=self.native_train_csv,
            parents=parents,
            seed=int(seed),
            torch_module=torch,
            dataset_name=self.dataset_name,
            native_train_parent_ids=self.native_train_parent_ids,
            native_train_payload=self.native_train_payload,
            num_classes=self.num_classes,
            source_label=self.source_label,
            frozen_gine_feature_schema=frozen_gine_feature_schema,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        official_api_signature_sha256 = _write_canonical_identity_once(
            output_dir / OFFICIAL_API_SIGNATURE_FILE,
            official_api_signature,
        )
        python_module_provenance_sha256 = _write_canonical_identity_once(
            output_dir / PYTHON_MODULE_PROVENANCE_FILE,
            python_module_provenance,
        )
        codec_summary = probe_source_graph_codec(
            source_dataset,
            parents,
            atom_attribute_audit_path=_source_atom_attribute_audit_path(
                output_dir
            ),
        )
        _write_json(output_dir / "source_codec_summary.json", codec_summary)
        require_source_codec_gate(codec_summary)
        use_frozen_gine = self.frozen_gine_checkpoint is not None
        if use_frozen_gine:
            frozen_gine_root = self.frozen_gine_checkpoint
            if frozen_gine_root is None or not (frozen_gine_root / "model.pt").is_file():
                raise FileNotFoundError(
                    "Frozen-GINE GlobalGCE bridge checkpoint is incomplete: "
                    f"{frozen_gine_root}"
                )
            gnn_checkpoint = frozen_gine_root / "model.pt"
        else:
            frozen_gine_root = None
            gnn_checkpoint = output_dir / "native_gnn.pt"
        model_checkpoint = output_dir / "globalgce_model.pt"
        rules_checkpoint = output_dir / "globalgce_rules.pt"
        frequent_subgraphs = output_dir / "frequent_subgraphs.pkl"
        training_state_path = output_dir / "training_core_summary.json"
        can_resume_trained_model = bool(
            resume
            and training_state_path.is_file()
            and gnn_checkpoint.is_file()
            and model_checkpoint.is_file()
            and rules_checkpoint.is_file()
        )
        training_state = (
            _read_json(training_state_path)
            if can_resume_trained_model
            else {}
        )
        exact_top_k_proof: dict[str, Any] = {}
        gspan_adoption_identity: dict[str, Any] = {}
        if gspan_adoption_proof is not None:
            from src.baselines.globalgce_mining_adoption import (
                validate_globalgce_gspan_adoption_proof,
            )

            gspan_adoption_identity.update(
                validate_globalgce_gspan_adoption_proof(gspan_adoption_proof)
            )
        if can_resume_trained_model:
            expected_state = {
                "dataset_name": self.dataset_name,
                "seed": int(seed),
                "epochs": int(epochs),
                "top_k_native": int(top_k_native),
                "learning_rate": float(learning_rate),
                "dropout": float(dropout),
                "gspan_exact_top_k_pruning": bool(
                    gspan_exact_top_k_pruning
                ),
                "gspan_adoption_identity": (
                    dict(gspan_adoption_identity)
                    if gspan_adoption_proof is not None
                    else None
                ),
                "selected_parent_count": len(parents),
                "num_classes": self.num_classes,
                "source_label": self.source_label,
                "target_label": self.target_label,
            }
            if use_frozen_gine:
                expected_state["prediction_backend"] = (
                    "frozen_gine_differentiable_bridge"
                )
            for key, expected in expected_state.items():
                if training_state.get(key) != expected:
                    raise ValueError(
                        "GlobalGCE trained-model resume configuration mismatch "
                        f"for {key}: actual={training_state.get(key)!r}, "
                        f"expected={expected!r}."
                    )
            for path, hash_key in (
                (gnn_checkpoint, "gnn_checkpoint_sha256"),
                (
                    model_checkpoint,
                    "globalgce_model_checkpoint_sha256",
                ),
                (rules_checkpoint, "rules_checkpoint_sha256"),
            ):
                actual_hash = _sha256_file(path)
                if training_state.get(hash_key) != actual_hash:
                    raise ValueError(
                        "GlobalGCE trained-model resume checkpoint hash "
                        f"mismatch for {path.name}: actual={actual_hash}, "
                        f"expected={training_state.get(hash_key)!r}."
                    )
            if gspan_exact_top_k_pruning:
                exact_top_k_proof.update(
                    validate_exact_top_k_proof_identity(
                        training_state.get("gspan_exact_top_k_proof") or {}
                    )
                )
            if gspan_adoption_proof is not None and training_state.get(
                "gspan_adoption_identity"
            ) != gspan_adoption_identity:
                raise ValueError(
                    "GlobalGCE trained-model resume adoption proof identity mismatch."
                )
        if use_frozen_gine:
            from src.baselines.globalgce_frozen_gine_bridge import (
                FrozenGINEDifferentiableBridge,
                GlobalGCEClassZeroTargetAdapter,
                GlobalGCETargetClassAdapter,
            )

            if self.frozen_gine_payloads is None:
                bridge = FrozenGINEDifferentiableBridge.from_checkpoint(
                    frozen_gine_root,
                    atom_symbols=tuple(source_dataset.atom_symbols),
                    bond_names=tuple(source_dataset.bond_names),
                    device=resolved_device,
                    expected_num_classes=self.num_classes,
                )
            else:
                bridge = FrozenGINEDifferentiableBridge.from_payloads(
                    self.frozen_gine_payloads,
                    atom_symbols=tuple(source_dataset.atom_symbols),
                    bond_names=tuple(source_dataset.bond_names),
                    device=resolved_device,
                    expected_num_classes=self.num_classes,
                )
            if (
                frozen_gine_feature_schema is None
                or bridge.feature_schema.to_dict()
                != frozen_gine_feature_schema.to_dict()
            ):
                raise GlobalGCEMutagenicityCodecError(
                    "Native GlobalGCE atom vocabulary and frozen GINE bridge "
                    "did not use the same feature schema."
                )
            legacy_bace_target_view = (
                self.num_classes == 2
                and self.source_label == 1
                and self.target_label == 0
            )
            if legacy_bace_target_view:
                gnn_model = GlobalGCEClassZeroTargetAdapter(bridge)
                gnn_summary = {
                    "prediction_backend": "frozen_gine_differentiable_bridge",
                    "classifier_family": "gine",
                    "oracle_backend": "gnn",
                    "rf_oracle_used": False,
                    "checkpoint_dir": str(frozen_gine_root),
                    "checkpoint_id": bridge.checkpoint_id,
                    "classifier_parameters_frozen": True,
                    "official_internal_target_label": 1,
                    "frozen_bace_source_label": self.source_label,
                    "frozen_bace_target_label": self.target_label,
                    "class_order_adapter": "official_1_maps_to_frozen_gine_0",
                }
            else:
                gnn_model = GlobalGCETargetClassAdapter(
                    bridge,
                    source_label=self.source_label,
                    target_label=self.target_label,
                )
                gnn_summary = {
                    "prediction_backend": "frozen_gine_differentiable_bridge",
                    "classifier_family": "gine",
                    "oracle_backend": "gnn",
                    "rf_oracle_used": False,
                    "checkpoint_dir": str(frozen_gine_root),
                    "checkpoint_id": bridge.checkpoint_id,
                    "classifier_parameters_frozen": True,
                    "official_internal_target_label": 1,
                    "num_classes": self.num_classes,
                    "frozen_source_label": self.source_label,
                    "frozen_target_label": self.target_label,
                    "class_order_adapter": (
                        "official_0_1_maps_to_frozen_source_target"
                    ),
                }
        else:
            native_train_loader = DataLoader(
                Subset(native_dataset, native_train_idx),
                batch_size=128,
                shuffle=True,
                generator=torch.Generator().manual_seed(int(seed)),
            )
            native_val_loader = DataLoader(
                Subset(native_dataset, native_val_idx),
                batch_size=128,
                shuffle=False,
            )
            gnn_model = modules["GTGNN"](
                native_dataset.node_feat_dim,
                32,
                2,
                native_dataset.edge_attr_dim,
                resolved_device,
                str(gnn_checkpoint),
            ).to(resolved_device)
            if can_resume_trained_model:
                gnn_model.load_state_dict(
                    torch.load(gnn_checkpoint, map_location=resolved_device)
                )
                gnn_summary = dict(training_state.get("gnn_training") or {})
            else:
                gnn_summary = _train_native_gnn(
                    gnn_model,
                    native_train_loader,
                    native_val_loader,
                    torch_module=torch,
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    checkpoint=gnn_checkpoint,
                )
            gnn_model.eval()
            for parameter in gnn_model.parameters():
                parameter.requires_grad_(False)

        if can_resume_trained_model:
            source_train_idx = [
                int(index)
                for index in training_state["source_train_idx"]
            ]
            source_val_idx = [
                int(index)
                for index in training_state["source_val_idx"]
            ]
            native_source_indices = set(
                source_train_idx + source_val_idx
            )
        elif use_frozen_gine:
            from src.data.molecular_graph_dataset import MolecularGraphData
            from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
            if self.frozen_gine_payloads is None:
                from src.oracles.oracle_factory import build_oracle

                oracle = build_oracle(
                    dataset=str(self.dataset_name).lower(),
                    backend="gnn",
                    checkpoint=frozen_gine_root,
                    device=resolved_device,
                    batch_size=256,
                    num_classes=self.num_classes,
                    source_label=self.source_label,
                )
            else:
                from src.oracles.gnn_oracle import GNNOracle

                oracle = GNNOracle.from_payloads(
                    self.frozen_gine_payloads,
                    device=resolved_device,
                    batch_size=256,
                    checkpoint_dir=frozen_gine_root,
                )
            featurizer = MolecularGraphFeaturizer(bridge.feature_schema)
            source_graphs = []
            for parent in parents:
                features = featurizer.featurize(parent.smiles)
                source_graphs.append(
                    MolecularGraphData(
                        x=features.node_features,
                        edge_index=features.edge_index,
                        edge_attr=features.edge_features,
                        y=self.source_label,
                        molecule_id=parent.parent_id,
                        smiles=features.canonical_smiles,
                        split="train",
                        graph_sha256=features.graph_sha256,
                    )
                )
            predictions = oracle.predict_records(source_graphs, batch_size=256)
            native_source_indices = {
                index
                for index, prediction in enumerate(predictions)
                if int(prediction["predicted_label"]) == self.source_label
            }
            source_train_idx = [
                index for index in source_train_idx if index in native_source_indices
            ]
            source_val_idx = [
                index for index in source_val_idx if index in native_source_indices
            ]
        else:
            source_prediction_loader = DataLoader(
                source_dataset,
                batch_size=500,
                shuffle=False,
            )
            native_source_indices = set()
            with torch.no_grad():
                for batch in source_prediction_loader:
                    predictions = gnn_model(
                        batch["feature"].to(resolved_device),
                        batch["adj"].to(resolved_device),
                        batch["edge_attr"].to(resolved_device),
                    )["y_pred"].argmax(-1).cpu()
                    native_source_indices.update(
                        int(index)
                        for index, prediction in zip(
                            batch["index"].tolist(),
                            predictions.tolist(),
                        )
                        if int(prediction) == 0
                    )
            source_train_idx = [
                index
                for index in source_train_idx
                if index in native_source_indices
            ]
            source_val_idx = [
                index
                for index in source_val_idx
                if index in native_source_indices
            ]
        if not source_train_idx or not source_val_idx:
            raise RuntimeError(
                "GlobalGCE classifier did not retain both internal train and "
                "validation source-label partitions."
            )
        training_resume_identity, training_resume_identity_sha256 = (
            _build_training_resume_identity(
                dataset_name=self.dataset_name,
                num_classes=self.num_classes,
                source_label=self.source_label,
                target_label=self.target_label,
                official_src=self.official_src,
                native_train_csv=self.native_train_csv,
                native_parents=native_parents,
                native_train_idx=native_train_idx,
                native_val_idx=native_val_idx,
                source_parents=parents,
                source_train_idx=source_train_idx,
                source_val_idx=source_val_idx,
                frozen_gine_checkpoint=frozen_gine_root,
                seed=int(seed),
                epochs=int(epochs),
                top_k_native=int(top_k_native),
                learning_rate=float(learning_rate),
                dropout=float(dropout),
                min_freq=self.min_freq,
                gspan_flush_every=int(gspan_flush_every),
                gspan_max_in_memory_candidates=int(
                    gspan_max_in_memory_candidates
                ),
                gspan_exact_top_k_pruning=bool(gspan_exact_top_k_pruning),
                gspan_adoption_identity=(
                    dict(gspan_adoption_identity)
                    if gspan_adoption_proof is not None
                    else None
                ),
            )
        )
        if can_resume_trained_model:
            for key, expected in (
                ("training_resume_identity", training_resume_identity),
                (
                    "training_resume_identity_sha256",
                    training_resume_identity_sha256,
                ),
            ):
                if training_state.get(key) != expected:
                    raise ValueError(
                        "GlobalGCE trained-model resume identity mismatch "
                        f"for {key}."
                    )
        source_dataset.train_idx = list(source_train_idx)
        source_dataset.val_idx = list(source_val_idx)
        source_dataset.test_idx = []
        source_train_loader = DataLoader(
            Subset(source_dataset, source_train_idx),
            batch_size=500,
            shuffle=False,
        )
        source_val_loader = DataLoader(
            Subset(source_dataset, source_val_idx),
            batch_size=500,
            shuffle=False,
        )
        globalgce_model = modules["GlobalGCE"](
            source_dataset.node_feat_dim,
            64,
            32,
            source_dataset.edge_attr_dim,
            float(dropout),
            3,
            20,
            int(top_k_native),
            False,
            str(frequent_subgraphs),
            resolved_device,
            gnn_model,
        ).to(resolved_device)
        if can_resume_trained_model:
            random.seed(int(seed))
            frozen_rules = torch.load(
                rules_checkpoint,
                map_location=resolved_device,
            )
            frozen_features = frozen_rules["feat"].detach().cpu()
            frozen_adjacencies = frozen_rules["adj"].detach().cpu()
            frozen_edges = (
                frozen_rules["edge_attr"].detach().cpu()
                if frozen_rules.get("edge_attr") is not None
                else None
            )
            official_utils = importlib.import_module("utils")
            frozen_subgraphs = {}
            for rule_index in range(len(frozen_features)):
                frozen_subgraphs[rule_index] = official_utils.get_nx_graph(
                    frozen_features[rule_index].argmax(-1),
                    frozen_adjacencies[rule_index],
                    (
                        frozen_edges[rule_index].argmax(-1)
                        if frozen_edges is not None
                        else None
                    ),
                )
            graph_sizes = [
                graph.number_of_nodes()
                for graph in frozen_subgraphs.values()
            ]
            globalgce_model.fsg.fs_max_nodes = max(graph_sizes)
            globalgce_model.fsg.fs_min_nodes = min(graph_sizes)
            augmented_dataset = globalgce_model.fsg.expand_data_by_fs(
                source_dataset,
                frozen_subgraphs,
            )
            globalgce_model.create_decoders()
            globalgce_model.load_state_dict(
                torch.load(model_checkpoint, map_location=resolved_device)
            )
            gnn_summary = dict(training_state.get("gnn_training") or {})
        else:
            def _record_exact_top_k_proof(proof: dict[str, Any]) -> None:
                exact_top_k_proof.clear()
                exact_top_k_proof.update(
                    validate_exact_top_k_proof_identity(proof)
                )

            def _record_gspan_adoption(proof: dict[str, Any]) -> None:
                if gspan_adoption_identity and proof != gspan_adoption_identity:
                    raise RuntimeError(
                        "GlobalGCE adoption identity changed between preflight and consumption."
                    )
                gspan_adoption_identity.clear()
                gspan_adoption_identity.update(proof)

            augmented_test_loader = train_globalgce_resumable(
                epochs=int(epochs),
                pred_model=gnn_model,
                model=globalgce_model,
                learning_rate=float(learning_rate),
                train_loader=source_train_loader,
                val_loader=source_val_loader,
                save_rule_path=rules_checkpoint,
                save_model_path=model_checkpoint,
                checkpoint_dir=output_dir / "globalgce_training_checkpoints",
                torch_module=torch,
                numpy_module=np,
                test_globalgce=modules["test_globalgce"],
                gspan_module=modules["gspan_module"],
                resume=bool(resume),
                gspan_flush_every=int(gspan_flush_every),
                gspan_max_in_memory_candidates=int(gspan_max_in_memory_candidates),
                gspan_exact_top_k_pruning=bool(gspan_exact_top_k_pruning),
                gspan_adoption_proof=gspan_adoption_proof,
                on_exact_top_k_proof=(
                    _record_exact_top_k_proof
                    if gspan_exact_top_k_pruning
                    else None
                ),
                on_gspan_adoption_proof=(
                    _record_gspan_adoption
                    if gspan_adoption_proof is not None
                    else None
                ),
                resume_identity=training_resume_identity,
                expected_resume_checkpoint=expected_resume_checkpoint,
                on_resume_checkpoint=on_resume_checkpoint,
                after_epoch_checkpoint=after_epoch_checkpoint,
            )
            augmented_dataset = augmented_test_loader.dataset.dataset
            if gspan_exact_top_k_pruning and not exact_top_k_proof:
                raise RuntimeError(
                    "GlobalGCE exact-top-k training did not bind its audit proof."
                )
            if gspan_adoption_proof is not None and not gspan_adoption_identity:
                raise RuntimeError(
                    "GlobalGCE training did not bind the adopted mining proof."
                )
            training_state_payload = {
                "dataset_name": self.dataset_name,
                "seed": int(seed),
                "epochs": int(epochs),
                "top_k_native": int(top_k_native),
                "learning_rate": float(learning_rate),
                "dropout": float(dropout),
                "gspan_exact_top_k_pruning": bool(gspan_exact_top_k_pruning),
                "gspan_adoption_identity": (
                    dict(gspan_adoption_identity)
                    if gspan_adoption_proof is not None
                    else None
                ),
                "selected_parent_count": len(parents),
                "num_classes": self.num_classes,
                "source_label": self.source_label,
                "target_label": self.target_label,
                "training_resume_identity": training_resume_identity,
                "training_resume_identity_sha256": (
                    training_resume_identity_sha256
                ),
                "source_train_idx": list(source_train_idx),
                "source_val_idx": list(source_val_idx),
                "gnn_training": gnn_summary,
                "gnn_checkpoint_sha256": _sha256_file(gnn_checkpoint),
                "globalgce_model_checkpoint_sha256": _sha256_file(
                    model_checkpoint
                ),
                "rules_checkpoint_sha256": _sha256_file(rules_checkpoint),
                "trained_once": True,
                "rule_selection_performed_once": True,
            }
            if gspan_exact_top_k_pruning:
                training_state_payload["gspan_exact_top_k_proof"] = dict(
                    exact_top_k_proof
                )
            if use_frozen_gine:
                training_state_payload["prediction_backend"] = (
                    "frozen_gine_differentiable_bridge"
                )
                training_state_payload["rf_oracle_used"] = False
            _write_json(training_state_path, training_state_payload)
        rules = torch.load(rules_checkpoint, map_location=resolved_device)
        rules = _detach_tensor_tree(rules)
        globalgce_model.eval()
        attach_globalgce_generation_dataset(
            augmented_dataset,
            source_dataset,
        )
        source_expansion_order = (
            list(source_dataset.train_idx)
            + list(source_dataset.val_idx)
            + list(source_dataset.test_idx)
        )
        codec_metadata = GlobalGCECodecMetadata.from_dataset(source_dataset)
        native_run_id = (
            f"{self.dataset_name.lower()}_seed{seed}_epochs{epochs}_topk{top_k_native}"
        )
        training_summary = {
            "official_entrypoints": [
                "data.data_preprocess-compatible dense molecular tensors",
                (
                    "project.FrozenGINEDifferentiableBridge"
                    if use_frozen_gine
                    else "models.GTGNN.GTGNN"
                ),
                "models.GlobalGCE.GlobalGCE",
                "models.models_utils.train_globalgce",
                "models.GlobalGCE.generate_cfs",
                "concate_inputs_with_local_recourse",
            ],
            "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
            "official_api_signature_sha256": (
                official_api_signature_sha256
            ),
            "python_module_provenance_sha256": (
                python_module_provenance_sha256
            ),
            "isolated_python": python_module_provenance["isolated_python"],
            "no_user_site": python_module_provenance["no_user_site"],
            "native_gnn_required": not use_frozen_gine,
            "prediction_backend": (
                "frozen_gine_differentiable_bridge"
                if use_frozen_gine
                else "official_native_gtgnn"
            ),
            "classifier_family": "gine" if use_frozen_gine else "gtgnn",
            "oracle_backend": "gnn",
            "rf_oracle_used": False if use_frozen_gine else None,
            "classifier_parameters_frozen": True,
            "num_classes": self.num_classes,
            "frozen_source_label": self.source_label,
            "frozen_target_label": self.target_label,
            "training_resume_identity": training_resume_identity,
            "training_resume_identity_sha256": training_resume_identity_sha256,
            "native_gnn_train_csv": str(self.native_train_csv),
            "native_gnn_train_rows": len(native_parents),
            "native_gnn_internal_train_ids_hash": _ids_hash(
                native_parents,
                native_train_idx,
            ),
            "native_gnn_internal_val_ids_hash": _ids_hash(
                native_parents,
                native_val_idx,
            ),
            "internal_train_ids_hash": _ids_hash(parents, source_train_idx),
            "internal_val_ids_hash": _ids_hash(parents, source_val_idx),
            "gnn_training": gnn_summary,
            "gnn_checkpoint": str(gnn_checkpoint.resolve()),
            "globalgce_model_checkpoint": str(model_checkpoint.resolve()),
            "rules_checkpoint": str(rules_checkpoint.resolve()),
            "frequent_subgraphs_path": str(frequent_subgraphs.resolve()),
            "raw_generated_rows": 0,
            "native_source_parent_count": len(native_source_indices),
            "generation_chunk_size": int(generation_chunk_size),
            "generation_num_workers": int(generation_num_workers),
            "generation_uses_inference_mode": True,
            "generation_requires_gradients": False,
            "trained_model_resumed": can_resume_trained_model,
            "training_checkpoint_policy": (
                "gspan_root_chunks_plus_epoch_atomic_identity_v2"
            ),
            "gspan_exact_top_k_pruning": bool(gspan_exact_top_k_pruning),
            "unused_pandas_report_materialization": False,
            **codec_summary,
            "saved_results_candidates_used": False,
            "generation_input_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
        }
        if self.num_classes == 2:
            training_summary.update(
                {
                    "frozen_bace_source_label": self.source_label,
                    "frozen_bace_target_label": self.target_label,
                }
            )
        if gspan_exact_top_k_pruning:
            training_summary["gspan_exact_top_k_proof"] = (
                validate_exact_top_k_proof_identity(exact_top_k_proof)
            )
        if gspan_adoption_proof is not None:
            training_summary["gspan_adoption_identity"] = dict(
                gspan_adoption_identity
            )
            training_summary["gspan_mining_route"] = (
                "user_scoped_read_only_v5_exhaustive_adoption"
            )
        else:
            training_summary["gspan_mining_route"] = (
                "fresh_exact_top_k" if gspan_exact_top_k_pruning else "fresh_exhaustive"
            )
        if use_frozen_gine:
            from src.baselines.globalgce_bace_native_rules import (
                GlobalGCENativeRule,
                stable_rule_id,
            )

            required_rule_tensors = (
                "feat",
                "adj",
                "edge_attr",
                "features_reconst",
                "adj_reconst",
                "edge_attrs_reconst",
            )
            missing_rule_tensors = [
                name for name in required_rule_tensors if rules.get(name) is None
            ]
            if missing_rule_tensors:
                raise RuntimeError(
                    "Frozen-GINE GlobalGCE rule checkpoint is incomplete: "
                    f"{missing_rule_tensors}"
                )
            rule_count = int(rules["feat"].shape[0])
            if any(int(rules[name].shape[0]) != rule_count for name in required_rule_tensors):
                raise RuntimeError("GlobalGCE frozen rule tensor counts differ")
            rule_rows: list[dict[str, Any]] = []
            rejected_rule_rows: list[dict[str, Any]] = []
            for rule_index in range(rule_count):
                raw_rule = {
                    "native_rule_index": rule_index,
                    "lhs_feature": rules["feat"][rule_index].detach().cpu().tolist(),
                    "lhs_adjacency": rules["adj"][rule_index].detach().cpu().tolist(),
                    "lhs_edge_attr": rules["edge_attr"][rule_index].detach().cpu().tolist(),
                    "rhs_feature": rules["features_reconst"][rule_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "rhs_adjacency": rules["adj_reconst"][rule_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "rhs_edge_attr": rules["edge_attrs_reconst"][rule_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "atom_symbols": list(source_dataset.atom_symbols),
                    "bond_names": list(source_dataset.bond_names),
                }
                candidate_id = stable_rule_id(raw_rule)
                raw_rule["rule_id"] = candidate_id
                row = {
                    "candidate_id": candidate_id,
                    "rank": rule_index + 1,
                    "native_rank": rule_index + 1,
                    "action_kind": "lhs_rhs_graph_transformation_rule",
                    "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
                    "oracle_backend": "gnn",
                    "classifier_family": "gine",
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": bridge.checkpoint_id,
                    "source_split": "train",
                    "rule": raw_rule,
                }
                try:
                    parsed_rule = GlobalGCENativeRule.from_payload(row)
                except Exception as exc:
                    rejected_rule_rows.append(
                        {
                            "native_rule_index": rule_index,
                            "candidate_id": candidate_id,
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                row["rule_content_hash"] = parsed_rule.content_hash()
                row["canonical_fragment"] = "N/A"
                row["canonical_fragment_reason"] = (
                    "GlobalGCE uses a native attachment-aware LHS-to-RHS rule"
                )
                row["selector_chemistry"] = parsed_rule.selector_chemistry()
                rule_rows.append(row)
            _write_jsonl(output_dir / "native_rule_catalog.jsonl", rule_rows)
            _write_jsonl(
                output_dir / "native_rule_rejections.jsonl", rejected_rule_rows
            )
            training_summary.update(
                {
                    "native_rule_count": rule_count,
                    "valid_native_rule_count": len(rule_rows),
                    "rejected_native_rule_count": len(rejected_rule_rows),
                    "native_rule_catalog": str(
                        (output_dir / "native_rule_catalog.jsonl").resolve()
                    ),
                    "native_rule_catalog_sha256": _sha256_file(
                        output_dir / "native_rule_catalog.jsonl"
                    ),
                    "native_rule_rejections": str(
                        (output_dir / "native_rule_rejections.jsonl").resolve()
                    ),
                }
            )
        log_globalgce_phase_memory(
            phase="training_and_rule_selection_ready",
            chunk_index=-1,
            processed_parent_count=0,
            raw_generated_count=0,
            torch_module=torch,
        )
        if on_training_ready is not None:
            on_training_ready(dict(training_summary))

        if rules_only:
            if not use_frozen_gine:
                raise ValueError("rules_only is reserved for the frozen-GINE route")
            if int(training_summary.get("valid_native_rule_count") or 0) < 20:
                raise RuntimeError(
                    "Frozen-GINE GlobalGCE produced fewer than twenty valid native rules"
                )
            _notify_generation_complete()
            return NativeGenerationResult([], training_summary)

        rows_by_parent = _augmented_rows_by_source_parent(
            augmented_dataset,
            source_expansion_order,
            len(parents),
        )
        all_records: list[dict[str, Any]] = []
        generated_this_call = 0
        total_chunks = math.ceil(len(parents) / int(generation_chunk_size))
        for parent_start in range(
            int(start_parent_offset),
            len(parents),
            int(generation_chunk_size),
        ):
            parent_end = min(
                len(parents),
                parent_start + int(generation_chunk_size),
            )
            chunk_index = parent_start // int(generation_chunk_size)
            augmented_row_indices = [
                row_index
                for parent_index in range(parent_start, parent_end)
                for row_index in rows_by_parent[parent_index]
            ]
            chunk_records: list[dict[str, Any]] = []
            if augmented_row_indices:
                chunk_dataset = _AugmentedGenerationChunk(
                    augmented_dataset,
                    source_dataset,
                    augmented_row_indices,
                )
                chunk_loader = DataLoader(
                    chunk_dataset,
                    batch_size=500,
                    shuffle=False,
                    num_workers=int(generation_num_workers),
                    persistent_workers=False,
                )
                validate_globalgce_generation_loader(chunk_loader)
                with torch.inference_mode():
                    cf_feat, cf_adj, cf_edge, graph_idx = modules[
                        "generate_cfs"
                    ](
                        chunk_loader,
                        rules,
                        resolved_device,
                    )
                ranks: dict[str, int] = defaultdict(int)
                for index in range(len(cf_feat)):
                    source_position = int(graph_idx[index])
                    if (
                        source_position < 0
                        or source_position >= len(source_expansion_order)
                    ):
                        raise RuntimeError(
                            "Official GlobalGCE returned invalid augmented "
                            f"source position {source_position}."
                        )
                    source_index = source_expansion_order[source_position]
                    parent = parents[source_index]
                    ranks[parent.parent_id] += 1
                    conversion = decode_globalgce_molecule(
                        cf_feat[index].detach(),
                        cf_adj[index].detach(),
                        (
                            cf_edge[index].detach()
                            if cf_edge is not None
                            else None
                        ),
                        metadata=codec_metadata,
                        graph_role="generated",
                        source_atom_attributes=(
                            source_dataset.source_atom_attributes[source_index]
                        ),
                    )
                    chunk_records.append(
                        {
                            "raw_smiles": conversion.smiles,
                            "source_parent_id": parent.parent_id,
                            "source_parent_smiles": parent.smiles,
                            "source_split": "train",
                            "generator_method": GENERATOR_METHOD,
                            "generator_rank": ranks[parent.parent_id],
                            "generator_score": -float(
                                ranks[parent.parent_id]
                            ),
                            "native_rule_id": (
                                "official_rule_application_"
                                f"{ranks[parent.parent_id]}"
                            ),
                            "native_run_id": native_run_id,
                            "native_conversion_ok": bool(conversion.ok),
                            "native_codec_decoded": bool(
                                conversion.codec_decoded
                            ),
                            "native_conversion_error_type": (
                                conversion.error_type
                            ),
                            "native_conversion_error": (
                                conversion.error_message
                            ),
                            "native_graph_record": conversion.graph_record,
                        }
                    )
                del cf_feat, cf_adj, cf_edge, graph_idx
                del chunk_loader, chunk_dataset
            generated_this_call += len(chunk_records)
            if on_chunk is not None:
                on_chunk(
                    chunk_index,
                    parent_start,
                    parent_end,
                    chunk_records,
                )
            else:
                all_records.extend(chunk_records)
            del chunk_records, augmented_row_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                chunk_index % int(memory_log_every_chunks) == 0
                or chunk_index + 1 == total_chunks
            ):
                log_globalgce_phase_memory(
                    phase="generation_chunk_complete",
                    chunk_index=chunk_index,
                    processed_parent_count=parent_end,
                    raw_generated_count=generated_this_call,
                    torch_module=torch,
                )
        training_summary["raw_generated_rows"] = generated_this_call
        training_summary["generation_chunk_count"] = max(
            0,
            total_chunks
            - (
                int(start_parent_offset)
                // int(generation_chunk_size)
            ),
        )
        _notify_generation_complete()
        return NativeGenerationResult(all_records, training_summary)


def _stratified_native_split(
    parents: Sequence[TrainParent],
    *,
    seed: int,
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    labels = sorted({parent.label for parent in parents})
    if len(labels) < 2:
        raise ValueError("Native GlobalGCE split requires at least two classes.")
    for label in labels:
        indices = [index for index, parent in enumerate(parents) if parent.label == label]
        if len(indices) < 2:
            raise ValueError(
                "Native GlobalGCE split requires two rows in every represented class."
            )
        indices.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}\t{parents[index].parent_id}".encode("utf-8")
            ).hexdigest()
        )
        val_count = max(1, int(round(0.20 * len(indices))))
        val.extend(indices[:val_count])
        train.extend(indices[val_count:])
    return sorted(train), sorted(val)


def _teacher_prediction(teacher: TeacherProtocol, smiles: str) -> tuple[int | None, bool]:
    result = teacher.score_smiles(smiles, label=TARGET_LABEL)
    ok = bool(result.get("teacher_result_ok"))
    prediction = result.get("teacher_label")
    return (int(prediction) if ok and prediction is not None else None), ok


def _annotate_and_filter_candidates(
    native_records: Sequence[dict[str, Any]],
    *,
    parents: Sequence[TrainParent],
    teacher: TeacherProtocol,
    seed: int,
    raw_index_offset: int = 0,
    dataset_name: str = DATASET_NAME,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    parent_by_id = {parent.parent_id: parent for parent in parents}
    raw_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    non_target_rows: list[dict[str, Any]] = []
    eligible_by_parent: dict[tuple[str, str], dict[str, Any]] = {}
    for local_raw_index, source in enumerate(native_records):
        raw_index = int(raw_index_offset) + local_raw_index
        parent_id = str(source.get("source_parent_id") or "").strip()
        if parent_id not in parent_by_id:
            raise ValueError(
                f"Official generator returned non-train source parent: {parent_id!r}"
            )
        parent = parent_by_id[parent_id]
        source_split = str(source.get("source_split") or "train").strip().lower()
        if source_split != "train":
            raise ValueError("Official generator returned a non-train source split.")
        raw_smiles = str(source.get("raw_smiles") or "").strip()
        canonical = _canonical_smiles(raw_smiles)
        molecule = Chem.MolFromSmiles(canonical) if canonical else None
        parse_ok = molecule is not None and molecule.GetNumAtoms() > 0
        native_conversion_ok = source.get("native_conversion_ok")
        if native_conversion_ok is not None:
            parse_ok = parse_ok and _bool_value(native_conversion_ok)
        native_error_type = str(
            source.get("native_conversion_error_type") or ""
        ).strip() or None
        native_error_message = str(
            source.get("native_conversion_error") or ""
        ).strip() or None
        teacher_pred: int | None = None
        teacher_ok = False
        if parse_ok and canonical is not None:
            teacher_pred, teacher_ok = _teacher_prediction(teacher, canonical)
        row = {
            "candidate_id": (
                stable_candidate_id(canonical, dataset_name=dataset_name)
                if canonical is not None
                else None
            ),
            "canonical_smiles": canonical,
            "raw_smiles": raw_smiles,
            "source_parent_id": parent.parent_id,
            "source_parent_smiles": parent.smiles,
            "source_split": "train",
            "generator_method": str(
                source.get("generator_method") or GENERATOR_METHOD
            ),
            "generator_rank": int(source.get("generator_rank") or raw_index + 1),
            "generator_score": _finite_float(source.get("generator_score")),
            "native_rule_id": source.get("native_rule_id"),
            "native_run_id": source.get("native_run_id"),
            "rdkit_parse_ok": bool(parse_ok),
            "native_codec_decoded": _bool_value(
                source.get("native_codec_decoded", parse_ok)
            ),
            "native_conversion_ok": _bool_value(
                source.get("native_conversion_ok", parse_ok)
            ),
            "native_conversion_error_type": native_error_type,
            "native_conversion_error": native_error_message,
            "teacher_pred": teacher_pred,
            "teacher_target_ok": bool(teacher_ok and teacher_pred == TARGET_LABEL),
            "num_atoms": int(molecule.GetNumAtoms()) if molecule is not None else 0,
            "num_bonds": int(molecule.GetNumBonds()) if molecule is not None else 0,
            "seed": int(seed),
            "raw_index": raw_index,
            "invalid_reason": (
                None
                if parse_ok
                else (
                    native_error_type
                    or "rdkit_parse_or_sanitize_failed"
                )
            ),
        }
        raw_rows.append(row)
        if not parse_ok or canonical is None:
            invalid_rows.append(dict(row))
            continue
        if not row["teacher_target_ok"]:
            non_target_rows.append(dict(row))
            continue
        key = (parent.parent_id, canonical)
        if key not in eligible_by_parent:
            eligible_by_parent[key] = dict(row)
    pool_rows = sorted(
        eligible_by_parent.values(),
        key=lambda row: (
            str(row["source_parent_id"]),
            int(row["generator_rank"]),
            str(row["canonical_smiles"]),
        ),
    )
    universe = _candidate_universe_from_pool(
        pool_rows,
        seed=seed,
        dataset_name=dataset_name,
    )
    return raw_rows, pool_rows, universe, invalid_rows, non_target_rows


def _deduplicate_candidate_pool(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible_by_parent: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("source_parent_id") or ""),
            str(row.get("canonical_smiles") or ""),
        )
        if key not in eligible_by_parent:
            eligible_by_parent[key] = dict(row)
    return sorted(
        eligible_by_parent.values(),
        key=lambda row: (
            str(row["source_parent_id"]),
            int(row["generator_rank"]),
            str(row["canonical_smiles"]),
        ),
    )


def _candidate_universe_from_pool(
    pool_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    dataset_name: str = DATASET_NAME,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        grouped[str(row["canonical_smiles"])].append(row)
    universe: list[dict[str, Any]] = []
    for canonical in sorted(grouped):
        occurrences = grouped[canonical]
        source_ids = sorted(
            {str(row["source_parent_id"]) for row in occurrences}
        )
        representative = occurrences[0]
        universe.append(
            {
                "candidate_id": stable_candidate_id(
                    canonical,
                    dataset_name=dataset_name,
                ),
                "canonical_smiles": canonical,
                "raw_smiles": representative["raw_smiles"],
                "generator_method": GENERATOR_METHOD,
                "teacher_pred": TARGET_LABEL,
                "teacher_target_ok": True,
                "rdkit_parse_ok": True,
                "num_atoms": representative["num_atoms"],
                "num_bonds": representative["num_bonds"],
                "source_parent_count": len(source_ids),
                "source_occurrence_count": len(occurrences),
                "source_parent_ids": source_ids,
                "seed": int(seed),
            }
        )
    return universe


def _raw_candidate_counts(path: Path) -> dict[str, int]:
    counts = {
        "raw_generated_rows": 0,
        "generated_codec_decoded_rows": 0,
        "generated_rdkit_valid_rows": 0,
        "teacher_target_rows": 0,
        "generated_invalid_valence_rows": 0,
        "generated_attribute_ambiguous_rows": 0,
        "generated_sanitize_failure_rows": 0,
    }
    for row in _iter_jsonl(path):
        counts["raw_generated_rows"] += 1
        counts["generated_codec_decoded_rows"] += int(
            _bool_value(row.get("native_codec_decoded"))
        )
        counts["generated_rdkit_valid_rows"] += int(
            _bool_value(row.get("rdkit_parse_ok"))
        )
        counts["teacher_target_rows"] += int(
            _bool_value(row.get("teacher_target_ok"))
        )
        reason = str(row.get("invalid_reason") or "")
        if reason == "generated_invalid_valence":
            counts["generated_invalid_valence_rows"] += 1
        elif reason == "generated_attribute_ambiguous":
            counts["generated_attribute_ambiguous_rows"] += 1
        elif reason == "generated_sanitize_failure":
            counts["generated_sanitize_failure_rows"] += 1
    return counts


def _config_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_mutagenicity_train_pool(
    *,
    train_csv: str | Path,
    teacher_path: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    teacher: TeacherProtocol,
    generator: NativeGeneratorProtocol,
    config: PoolBuildConfig | None = None,
    dataset_name: str = DATASET_NAME,
) -> dict[str, Any]:
    resolved = config or PoolBuildConfig()
    if int(resolved.epochs) <= 0 or int(resolved.top_k_native) <= 0:
        raise ValueError("epochs and top_k_native must be positive.")
    if not 0.0 <= float(resolved.dropout) < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if float(resolved.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if int(resolved.generation_chunk_size) <= 0:
        raise ValueError("generation_chunk_size must be positive.")
    if int(resolved.generation_num_workers) < 0:
        raise ValueError("generation_num_workers must be non-negative.")
    if int(resolved.memory_log_every_chunks) <= 0:
        raise ValueError("memory_log_every_chunks must be positive.")
    if (
        int(resolved.gspan_flush_every) <= 0
        or int(resolved.gspan_max_in_memory_candidates) <= 0
    ):
        raise ValueError("gSpan streaming limits must be positive.")
    train_path = Path(train_csv).expanduser().resolve()
    teacher_file = Path(teacher_path).expanduser().resolve()
    official_src = _resolve_official_src(official_root)
    all_parents, selected_parents = load_strict_train_parents(
        train_path,
        parent_limit=int(resolved.parent_limit),
        expected_parent_count=int(resolved.expected_parent_count),
        forbid_calibration_test=bool(resolved.forbid_calibration_test),
    )
    cohort_hash = train_cohort_hash(all_parents)
    selected_hash = train_cohort_hash(selected_parents)
    destination = Path(output_dir).expanduser().resolve()
    complete_path = destination / "_RUN_COMPLETE.json"
    if complete_path.is_file():
        raise FileExistsError(f"Completed GlobalGCE train pool cannot be rerun: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        "dataset_name": str(dataset_name),
        "train_csv": _file_identity(train_path),
        "teacher_path": _file_identity(teacher_file),
        "official_root": _file_identity(official_src),
        "train_cohort_hash": cohort_hash,
        "selected_train_cohort_hash": selected_hash,
        "parent_limit": int(resolved.parent_limit),
        "seed": int(resolved.seed),
        "epochs": int(resolved.epochs),
        "top_k_native": int(resolved.top_k_native),
        "learning_rate": float(resolved.learning_rate),
        "dropout": float(resolved.dropout),
        "device": str(resolved.device),
        "forbid_calibration_test": bool(resolved.forbid_calibration_test),
        "generation_chunk_size": int(resolved.generation_chunk_size),
        "generation_num_workers": int(resolved.generation_num_workers),
        "memory_log_every_chunks": int(
            resolved.memory_log_every_chunks
        ),
        "gspan_flush_every": int(resolved.gspan_flush_every),
        "gspan_max_in_memory_candidates": int(
            resolved.gspan_max_in_memory_candidates
        ),
        "gspan_exact_top_k_pruning": bool(
            resolved.gspan_exact_top_k_pruning
        ),
        "native_generator": (
            generator.config_identity()
            if callable(getattr(generator, "config_identity", None))
            else {
                "generator_class": (
                    f"{type(generator).__module__}.{type(generator).__qualname__}"
                )
            }
        ),
    }
    fingerprint = _config_fingerprint(fingerprint_payload)
    manifest_path = destination / "run_manifest.json"
    checkpoint_path = destination / "resume_checkpoint.json"
    existing = list(destination.iterdir())
    if existing and not resolved.resume:
        raise FileExistsError("Output directory is non-empty and resume is disabled.")
    if existing:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise ValueError("Resume requires run_manifest.json and resume_checkpoint.json.")
        manifest = _read_json(manifest_path)
        checkpoint = _read_json(checkpoint_path)
        if manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration mismatch in run_manifest.json.")
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration mismatch in resume_checkpoint.json.")
    else:
        manifest = {
            "created_at": _utc_now(),
            "config_fingerprint": fingerprint,
            "inputs": fingerprint_payload,
            "dataset": str(dataset_name),
            "source_label": SOURCE_LABEL,
            "target_label": TARGET_LABEL,
            "generation_input_split": "train",
            "candidate_selection_performed": False,
            "calibration_used": False,
            "test_used": False,
            "teacher_used_only_for_target_validation": True,
            "official_saved_results_candidates_used": False,
            "run_complete": False,
        }
        _write_json(manifest_path, manifest)
        _write_json(
            checkpoint_path,
            {
                "config_fingerprint": fingerprint,
                "stage": "initialized",
                "run_complete": False,
                "updated_at": _utc_now(),
            },
        )

    raw_path = destination / "raw_generated_candidates.jsonl"
    invalid_path = destination / "invalid_candidates.jsonl"
    non_target_path = destination / "non_target_candidates.jsonl"
    pool_path = destination / "candidate_pool.jsonl"
    part_paths = {
        "raw": raw_path.with_suffix(raw_path.suffix + ".part"),
        "invalid": invalid_path.with_suffix(invalid_path.suffix + ".part"),
        "non_target": non_target_path.with_suffix(
            non_target_path.suffix + ".part"
        ),
        "pool": pool_path.with_suffix(pool_path.suffix + ".part"),
    }
    generation_checkpoint_path = (
        destination / "generation_resume_checkpoint.json"
    )
    training_path = destination / "training_summary.json"
    generation_state: dict[str, Any]
    if generation_checkpoint_path.is_file():
        generation_state = _read_json(generation_checkpoint_path)
        expected_resume = {
            "config_hash": fingerprint,
            "train_cohort_hash": cohort_hash,
            "selected_cohort_hash": selected_hash,
            "generation_chunk_size": int(resolved.generation_chunk_size),
        }
        for key, expected in expected_resume.items():
            if generation_state.get(key) != expected:
                raise ValueError(
                    "Generation resume configuration/hash mismatch for "
                    f"{key}: actual={generation_state.get(key)!r}, "
                    f"expected={expected!r}."
                )
        recorded_model_hash = generation_state.get(
            "model_checkpoint_hash"
        )
        model_checkpoint = (
            destination / "native" / "globalgce_model.pt"
        )
        if recorded_model_hash is not None:
            if not model_checkpoint.is_file():
                raise ValueError(
                    "Generation resume checkpoint records a trained model hash "
                    f"but the checkpoint is missing: {model_checkpoint}"
                )
            actual_model_hash = _sha256_file(model_checkpoint)
            if actual_model_hash != recorded_model_hash:
                raise ValueError(
                    "Generation resume model checkpoint hash mismatch: "
                    f"actual={actual_model_hash}, "
                    f"expected={recorded_model_hash}."
                )
        for name, final_path in (
            ("raw", raw_path),
            ("invalid", invalid_path),
            ("non_target", non_target_path),
            ("pool", pool_path),
        ):
            if not part_paths[name].exists() and final_path.exists():
                os.replace(final_path, part_paths[name])
        _truncate_jsonl(
            part_paths["raw"],
            int(generation_state.get("raw_generated_rows") or 0),
        )
        _truncate_jsonl(
            part_paths["invalid"],
            int(generation_state.get("invalid_rows") or 0),
        )
        _truncate_jsonl(
            part_paths["non_target"],
            int(generation_state.get("non_target_rows") or 0),
        )
        _truncate_jsonl(
            part_paths["pool"],
            int(generation_state.get("candidate_pool_rows") or 0),
        )
    else:
        stale_parts = [
            str(path)
            for path in part_paths.values()
            if path.exists() and path.stat().st_size > 0
        ]
        if stale_parts:
            raise ValueError(
                "Generation part files exist without a valid "
                f"generation_resume_checkpoint.json: {stale_parts}"
            )
        for path in part_paths.values():
            _write_jsonl(path, [])
        generation_state = {
            "config_hash": fingerprint,
            "train_cohort_hash": cohort_hash,
            "selected_cohort_hash": selected_hash,
            "model_checkpoint_hash": None,
            "completed_chunk_count": 0,
            "next_parent_offset": 0,
            "raw_generated_rows": 0,
            "invalid_rows": 0,
            "non_target_rows": 0,
            "candidate_pool_rows": 0,
            "git_commit": _git_commit(),
            "generation_chunk_size": int(resolved.generation_chunk_size),
            "run_complete": False,
            "updated_at": _utc_now(),
        }
        _write_json(generation_checkpoint_path, generation_state)

    start_parent_offset = int(
        generation_state.get("next_parent_offset") or 0
    )
    if start_parent_offset < 0 or start_parent_offset > len(selected_parents):
        raise ValueError(
            "Generation resume next_parent_offset is outside the selected "
            f"cohort: {start_parent_offset}."
        )
    training_summary_holder: dict[str, Any] = (
        _read_json(training_path) if training_path.is_file() else {}
    )
    chunks_consumed = 0

    def _model_checkpoint_hash() -> str | None:
        model_path = destination / "native" / "globalgce_model.pt"
        return _sha256_file(model_path) if model_path.is_file() else None

    def _training_ready(summary_payload: dict[str, Any]) -> None:
        training_summary_holder.clear()
        training_summary_holder.update(summary_payload)
        _write_json(training_path, training_summary_holder)
        generation_state["model_checkpoint_hash"] = (
            _model_checkpoint_hash()
        )
        generation_state["updated_at"] = _utc_now()
        _write_json(generation_checkpoint_path, generation_state)

    def _consume_chunk(
        chunk_index: int,
        parent_start: int,
        parent_end: int,
        native_records: list[dict[str, Any]],
    ) -> None:
        nonlocal chunks_consumed
        expected_start = int(
            generation_state.get("next_parent_offset") or 0
        )
        if int(parent_start) != expected_start:
            raise RuntimeError(
                "Generation chunk parent offset mismatch: "
                f"expected={expected_start}, actual={parent_start}."
            )
        (
            raw_chunk,
            pool_chunk,
            _universe_chunk,
            invalid_chunk,
            non_target_chunk,
        ) = _annotate_and_filter_candidates(
            native_records,
            parents=selected_parents,
            teacher=teacher,
            seed=int(resolved.seed),
            raw_index_offset=int(
                generation_state.get("raw_generated_rows") or 0
            ),
            dataset_name=str(dataset_name),
        )
        _append_jsonl(part_paths["raw"], raw_chunk)
        _append_jsonl(part_paths["invalid"], invalid_chunk)
        _append_jsonl(part_paths["non_target"], non_target_chunk)
        _append_jsonl(part_paths["pool"], pool_chunk)
        generation_state.update(
            {
                "model_checkpoint_hash": (
                    _model_checkpoint_hash()
                    or generation_state.get("model_checkpoint_hash")
                ),
                "completed_chunk_count": int(
                    generation_state.get("completed_chunk_count") or 0
                )
                + 1,
                "next_parent_offset": int(parent_end),
                "raw_generated_rows": int(
                    generation_state.get("raw_generated_rows") or 0
                )
                + len(raw_chunk),
                "invalid_rows": int(
                    generation_state.get("invalid_rows") or 0
                )
                + len(invalid_chunk),
                "non_target_rows": int(
                    generation_state.get("non_target_rows") or 0
                )
                + len(non_target_chunk),
                "candidate_pool_rows": int(
                    generation_state.get("candidate_pool_rows") or 0
                )
                + len(pool_chunk),
                "last_chunk_index": int(chunk_index),
                "run_complete": False,
                "updated_at": _utc_now(),
            }
        )
        _write_json(generation_checkpoint_path, generation_state)
        _write_json(
            checkpoint_path,
            {
                "config_fingerprint": fingerprint,
                "stage": "generating_chunks",
                "next_parent_offset": int(parent_end),
                "raw_generated_rows": generation_state[
                    "raw_generated_rows"
                ],
                "run_complete": False,
                "updated_at": _utc_now(),
            },
        )
        chunks_consumed += 1
        if (
            int(chunk_index) % int(resolved.memory_log_every_chunks) == 0
            or int(parent_end) == len(selected_parents)
        ):
            log_globalgce_phase_memory(
                phase="chunk_annotated_and_flushed",
                chunk_index=int(chunk_index),
                processed_parent_count=int(parent_end),
                raw_generated_count=int(
                    generation_state["raw_generated_rows"]
                ),
            )
        del raw_chunk, pool_chunk, invalid_chunk, non_target_chunk

    result: NativeGenerationResult
    if start_parent_offset < len(selected_parents):
        result = generator.generate(
            selected_parents,
            output_dir=destination / "native",
            seed=int(resolved.seed),
            epochs=int(resolved.epochs),
            top_k_native=int(resolved.top_k_native),
            learning_rate=float(resolved.learning_rate),
            dropout=float(resolved.dropout),
            device=str(resolved.device),
            resume=bool(resolved.resume),
            generation_chunk_size=int(resolved.generation_chunk_size),
            generation_num_workers=int(resolved.generation_num_workers),
            memory_log_every_chunks=int(
                resolved.memory_log_every_chunks
            ),
            gspan_flush_every=int(resolved.gspan_flush_every),
            gspan_max_in_memory_candidates=int(
                resolved.gspan_max_in_memory_candidates
            ),
            **(
                {"gspan_exact_top_k_pruning": True}
                if resolved.gspan_exact_top_k_pruning
                else {}
            ),
            start_parent_offset=start_parent_offset,
            on_training_ready=_training_ready,
            on_chunk=_consume_chunk,
        )
        training_summary_holder.update(result.training_summary)
        if result.records:
            if start_parent_offset != 0 or chunks_consumed:
                raise RuntimeError(
                    "Legacy non-streaming generator returned records after "
                    "chunked generation had already started."
                )
            _consume_chunk(
                0,
                0,
                len(selected_parents),
                list(result.records),
            )
        elif chunks_consumed == 0:
            _consume_chunk(
                start_parent_offset
                // int(resolved.generation_chunk_size),
                start_parent_offset,
                len(selected_parents),
                [],
            )
        _write_json(training_path, training_summary_holder)
    elif not training_summary_holder:
        native_training_state = (
            destination / "native" / "training_core_summary.json"
        )
        if not native_training_state.is_file():
            raise ValueError(
                "Generation rows are complete but training_summary.json is "
                "missing, and no native training state is available."
            )
        training_summary_holder.update(_read_json(native_training_state))
        _write_json(training_path, training_summary_holder)

    if int(generation_state.get("next_parent_offset") or 0) != len(
        selected_parents
    ):
        raise RuntimeError(
            "Chunked generation ended before every selected parent was "
            f"processed: next={generation_state.get('next_parent_offset')}, "
            f"expected={len(selected_parents)}."
        )
    training_summary = dict(training_summary_holder)
    training_summary["raw_generated_rows"] = int(
        generation_state.get("raw_generated_rows") or 0
    )
    _write_json(training_path, training_summary)

    for name, final_path in (
        ("raw", raw_path),
        ("invalid", invalid_path),
        ("non_target", non_target_path),
    ):
        os.replace(part_paths[name], final_path)
    pool_rows = _deduplicate_candidate_pool(
        _iter_jsonl(part_paths["pool"])
    )
    universe = _candidate_universe_from_pool(
        pool_rows,
        seed=int(resolved.seed),
        dataset_name=str(dataset_name),
    )
    _write_jsonl(pool_path, pool_rows)
    _write_jsonl(destination / "candidate_universe.jsonl", universe)
    part_paths["pool"].unlink(missing_ok=True)
    source_parent_ids = {str(row["source_parent_id"]) for row in pool_rows}
    raw_counts = _raw_candidate_counts(raw_path)
    raw_generated_rows = raw_counts["raw_generated_rows"]
    generated_valid_rows = raw_counts["generated_rdkit_valid_rows"]
    generated_decoded_rows = raw_counts["generated_codec_decoded_rows"]
    generated_invalid_valence_rows = raw_counts[
        "generated_invalid_valence_rows"
    ]
    generated_attribute_ambiguous_rows = raw_counts[
        "generated_attribute_ambiguous_rows"
    ]
    generated_sanitize_failure_rows = raw_counts[
        "generated_sanitize_failure_rows"
    ]
    generated_invalid_other_rows = (
        raw_generated_rows
        - generated_valid_rows
        - generated_invalid_valence_rows
    )
    source_codec_passed = training_summary.get("source_codec_passed")
    pool_nonempty = len(pool_rows) > 0 and len(universe) > 0
    summary = {
        "input_train_rows": len(all_parents),
        "selected_train_rows": len(selected_parents),
        "unique_source_parents": len(source_parent_ids),
        "raw_generated_rows": raw_generated_rows,
        "rdkit_valid_rows": generated_valid_rows,
        "teacher_target_rows": raw_counts["teacher_target_rows"],
        "candidate_pool_rows": len(pool_rows),
        "canonical_unique_candidates": len(universe),
        "source_parent_coverage": (
            len(source_parent_ids) / len(selected_parents) if selected_parents else 0.0
        ),
        "train_cohort_hash": cohort_hash,
        "selected_train_cohort_hash": selected_hash,
        "internal_train_ids_hash": training_summary.get("internal_train_ids_hash"),
        "internal_val_ids_hash": training_summary.get("internal_val_ids_hash"),
        "source_codec_checked_rows": int(
            training_summary.get("source_codec_checked_rows") or 0
        ),
        "source_codec_rdkit_valid_rows": int(
            training_summary.get("source_codec_rdkit_valid_rows") or 0
        ),
        "source_codec_structure_match_rows": int(
            training_summary.get("source_codec_structure_match_rows") or 0
        ),
        "source_codec_invalid_valence_rows": int(
            training_summary.get("source_codec_invalid_valence_rows") or 0
        ),
        "source_codec_attribute_mapping_failed_rows": int(
            training_summary.get(
                "source_codec_attribute_mapping_failed_rows"
            )
            or 0
        ),
        "source_codec_passed": source_codec_passed,
        "source_codec_failure_examples": list(
            training_summary.get("source_codec_failure_examples") or []
        ),
        "atom_attribute_source": training_summary.get(
            "atom_attribute_source"
        ),
        "formal_charge_encoded_by_native": training_summary.get(
            "formal_charge_encoded_by_native"
        ),
        "source_atom_mapping_method": training_summary.get(
            "source_atom_mapping_method"
        ),
        "source_atom_mapping_unique": training_summary.get(
            "source_atom_mapping_unique"
        ),
        "source_formal_charge_nonzero_atom_count": int(
            training_summary.get("source_formal_charge_nonzero_atom_count")
            or 0
        ),
        "generated_graph_rows": raw_generated_rows,
        "generated_codec_decoded_rows": generated_decoded_rows,
        "generated_rdkit_valid_rows": generated_valid_rows,
        "generated_invalid_valence_rows": generated_invalid_valence_rows,
        "generated_attribute_ambiguous_rows": (
            generated_attribute_ambiguous_rows
        ),
        "generated_sanitize_failure_rows": (
            generated_sanitize_failure_rows
        ),
        "generated_invalid_other_rows": generated_invalid_other_rows,
        "generated_valid_rate": (
            generated_valid_rows / raw_generated_rows
            if raw_generated_rows
            else 0.0
        ),
        "generation_chunk_size": int(resolved.generation_chunk_size),
        "generation_num_workers": int(resolved.generation_num_workers),
        "generation_completed_chunks": int(
            generation_state.get("completed_chunk_count") or 0
        ),
        "generation_peak_materialization": "one_parent_chunk",
        "generation_uses_inference_mode": bool(
            training_summary.get("generation_uses_inference_mode", False)
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": pool_nonempty,
    }
    _write_json(destination / "summary.json", summary)
    generation_state.update(
        {
            "model_checkpoint_hash": (
                _model_checkpoint_hash()
                or generation_state.get("model_checkpoint_hash")
            ),
            "next_parent_offset": len(selected_parents),
            "raw_generated_rows": raw_generated_rows,
            "candidate_pool_rows": len(pool_rows),
            "generation_complete": True,
            "run_complete": bool(pool_nonempty),
            "updated_at": _utc_now(),
        }
    )
    _write_json(generation_checkpoint_path, generation_state)
    if not pool_nonempty:
        failure = {
            "error": "empty_candidate_universe",
            "source_codec_passed": source_codec_passed,
            "raw_generated_rows": raw_generated_rows,
            "generated_codec_decoded_rows": generated_decoded_rows,
            "generated_rdkit_valid_rows": generated_valid_rows,
            "generated_invalid_valence_rows": generated_invalid_valence_rows,
            "generated_attribute_ambiguous_rows": (
                generated_attribute_ambiguous_rows
            ),
            "generated_sanitize_failure_rows": (
                generated_sanitize_failure_rows
            ),
            "generated_invalid_other_rows": generated_invalid_other_rows,
            "candidate_pool_rows": len(pool_rows),
            "canonical_unique_candidates": len(universe),
        }
        manifest.update(
            {
                "run_complete": False,
                "failed_at": _utc_now(),
                "failure": failure,
                "training_summary": str(training_path.resolve()),
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(
            checkpoint_path,
            {
                "config_fingerprint": fingerprint,
                "stage": "failed_empty_candidate_universe",
                "run_complete": False,
                **failure,
                "updated_at": _utc_now(),
            },
        )
        raise GlobalGCEEmptyCandidateUniverseError(
            "GlobalGCE generated no eligible Mutagenicity candidates. "
            f"source_codec_passed={source_codec_passed!r}, "
            f"raw_generated_rows={raw_generated_rows}, "
            f"generated_codec_decoded_rows={generated_decoded_rows}, "
            f"generated_rdkit_valid_rows={generated_valid_rows}, "
            f"generated_invalid_valence_rows={generated_invalid_valence_rows}, "
            "generated_attribute_ambiguous_rows="
            f"{generated_attribute_ambiguous_rows}, "
            "generated_sanitize_failure_rows="
            f"{generated_sanitize_failure_rows}, "
            f"generated_invalid_other_rows={generated_invalid_other_rows}, "
            f"candidate_pool_rows={len(pool_rows)}, "
            f"canonical_unique_candidates={len(universe)}."
        )
    manifest.update(
        {
            "run_complete": True,
            "completed_at": _utc_now(),
            "training_summary": str(training_path.resolve()),
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        checkpoint_path,
        {
            "config_fingerprint": fingerprint,
            "stage": "complete",
            "raw_generated_rows": raw_generated_rows,
            "candidate_pool_rows": len(pool_rows),
            "run_complete": True,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        complete_path,
        {
            "run_complete": True,
            "config_fingerprint": fingerprint,
            "completed_at": _utc_now(),
            "candidate_pool_rows": len(pool_rows),
            "canonical_unique_candidates": len(universe),
        },
    )
    return summary


def audit_mutagenicity_train_pool(
    run_dir: str | Path,
    *,
    train_csv: str | Path,
    expected_parent_count: int = DEFAULT_EXPECTED_PARENT_COUNT,
    expected_input_train_count: int | None = None,
    require_target_label_zero: bool = True,
    require_unique_universe: bool = True,
    forbid_calibration_test: bool = True,
    require_complete: bool = True,
    dataset_name: str = DATASET_NAME,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    for relative in REQUIRED_OUTPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing GlobalGCE train-pool artifact: {path}")
        if (
            relative
            not in {"invalid_candidates.jsonl", "non_target_candidates.jsonl"}
            and path.stat().st_size <= 0
        ):
            raise FileNotFoundError(f"Empty GlobalGCE train-pool artifact: {path}")
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "run_manifest.json")
    if str(manifest.get("dataset") or "") != str(dataset_name):
        raise AssertionError(
            "GlobalGCE train-pool dataset identity mismatch: "
            f"actual={manifest.get('dataset')!r}, expected={dataset_name!r}."
        )
    manifest_inputs = dict(manifest.get("inputs") or {})
    parent_limit = int(
        manifest_inputs.get("parent_limit") or 0
    )
    if parent_limit < 0:
        raise AssertionError(
            f"Manifest parent_limit must be non-negative, got {parent_limit}."
        )
    selection_seed = int(manifest_inputs.get("seed") or 0)
    selection_method = str(
        manifest_inputs.get("deterministic_parent_selection_method")
        or "parent_id_ascending_prefix_v1"
    )
    parents, selected_parents = load_strict_train_parents(
        train_csv,
        parent_limit=parent_limit,
        expected_parent_count=(
            int(expected_input_train_count)
            if expected_input_train_count is not None
            else 0
        ),
        forbid_calibration_test=bool(forbid_calibration_test),
    )
    if int(expected_parent_count) > 0 and len(selected_parents) != int(
        expected_parent_count
    ):
        raise ValueError(
            "Selected strict train parent count mismatch: "
            f"expected={expected_parent_count}, "
            f"found={len(selected_parents)}, "
            f"input_train_rows={len(parents)}, parent_limit={parent_limit}."
        )
    if int(summary.get("selected_train_rows") or -1) != len(selected_parents):
        raise AssertionError(
            "Summary selected_train_rows does not match the cohort rebuilt "
            "with load_strict_train_parents: "
            f"summary={summary.get('selected_train_rows')!r}, "
            f"rebuilt={len(selected_parents)}."
        )
    full_parent_ids = {parent.parent_id for parent in parents}
    selected_parent_ids = {parent.parent_id for parent in selected_parents}
    rebuilt_selected_hash = train_cohort_hash(selected_parents)
    selected_hash_matches = (
        summary.get("selected_train_cohort_hash")
        == rebuilt_selected_hash
    )
    if not selected_hash_matches:
        raise AssertionError(
            "Selected train cohort hash mismatch: "
            f"summary={summary.get('selected_train_cohort_hash')!r}, "
            f"rebuilt={rebuilt_selected_hash!r}."
        )
    raw_rows = _read_jsonl(root / "raw_generated_candidates.jsonl")
    pool_rows = _read_jsonl(root / "candidate_pool.jsonl")
    universe = _read_jsonl(root / "candidate_universe.jsonl")
    invalid = _read_jsonl(root / "invalid_candidates.jsonl")
    non_target = _read_jsonl(root / "non_target_candidates.jsonl")
    for row in raw_rows:
        parent_id = str(row.get("source_parent_id") or "")
        if parent_id not in full_parent_ids:
            raise AssertionError("Raw candidate references a non-train parent.")
        if parent_id not in selected_parent_ids:
            raise AssertionError(
                "Raw candidate references a train parent outside the "
                "deterministically selected cohort."
            )
        if str(row.get("source_split") or "").lower() != "train":
            raise AssertionError("Raw candidate source_split is not train.")
    for row in pool_rows:
        parent_id = str(row.get("source_parent_id") or "")
        if parent_id not in full_parent_ids:
            raise AssertionError("Candidate pool references a non-train parent.")
        if parent_id not in selected_parent_ids:
            raise AssertionError(
                "Candidate pool references a train parent outside the "
                "deterministically selected cohort."
            )
        if str(row.get("source_split") or "").lower() != "train":
            raise AssertionError("Candidate pool source_split is not train.")
        canonical = str(row.get("canonical_smiles") or "")
        if stable_candidate_id(
            canonical,
            dataset_name=dataset_name,
        ) != row.get("candidate_id"):
            raise AssertionError("Candidate pool stable ID mismatch.")
        if require_target_label_zero and (
            int(row.get("teacher_pred")) != TARGET_LABEL
            or not _bool_value(row.get("teacher_target_ok"))
        ):
            raise AssertionError("Candidate pool contains a non-target molecule.")
    pool_keys = [
        (str(row["source_parent_id"]), str(row["canonical_smiles"]))
        for row in pool_rows
    ]
    if len(pool_keys) != len(set(pool_keys)):
        raise AssertionError("Candidate pool contains within-parent duplicates.")
    universe_smiles = [str(row.get("canonical_smiles") or "") for row in universe]
    if require_unique_universe and len(universe_smiles) != len(set(universe_smiles)):
        raise AssertionError("Candidate universe contains duplicate canonical SMILES.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        grouped[str(row["canonical_smiles"])].append(row)
    for row in universe:
        canonical = str(row.get("canonical_smiles") or "")
        if _canonical_smiles(canonical) != canonical:
            raise AssertionError("Candidate universe contains invalid/noncanonical SMILES.")
        if stable_candidate_id(
            canonical,
            dataset_name=dataset_name,
        ) != row.get("candidate_id"):
            raise AssertionError("Candidate universe stable ID mismatch.")
        if int(row.get("teacher_pred")) != TARGET_LABEL:
            raise AssertionError("Candidate universe contains teacher_pred != 0.")
        expected_sources = {
            str(item["source_parent_id"]) for item in grouped.get(canonical, [])
        }
        if int(row.get("source_parent_count") or 0) != len(expected_sources):
            raise AssertionError("Candidate universe source_parent_count mismatch.")
        if int(row.get("source_occurrence_count") or 0) != len(
            grouped.get(canonical, [])
        ):
            raise AssertionError("Candidate universe source occurrence mismatch.")
    complete = _read_json(root / "_RUN_COMPLETE.json")
    unique_source_parents = {
        str(row["source_parent_id"]) for row in pool_rows
    }
    candidate_sources_subset = unique_source_parents <= selected_parent_ids
    if not candidate_sources_subset:
        raise AssertionError(
            "Candidate source parents are not a subset of the selected cohort."
        )
    source_parent_coverage_recomputed = (
        len(unique_source_parents) / len(selected_parents)
        if selected_parents
        else 0.0
    )
    recalculated = {
        "input_train_rows": len(parents),
        "selected_train_rows": len(selected_parents),
        "unique_source_parents": len(unique_source_parents),
        "raw_generated_rows": len(raw_rows),
        "rdkit_valid_rows": sum(_bool_value(row.get("rdkit_parse_ok")) for row in raw_rows),
        "teacher_target_rows": sum(
            _bool_value(row.get("teacher_target_ok")) for row in raw_rows
        ),
        "candidate_pool_rows": len(pool_rows),
        "canonical_unique_candidates": len(universe),
        "train_cohort_hash": train_cohort_hash(parents),
        "selected_train_cohort_hash": train_cohort_hash(selected_parents),
        "source_parent_coverage": (
            source_parent_coverage_recomputed
        ),
        "generated_graph_rows": len(raw_rows),
        "generated_codec_decoded_rows": sum(
            _bool_value(row.get("native_codec_decoded")) for row in raw_rows
        ),
        "generated_rdkit_valid_rows": sum(
            _bool_value(row.get("rdkit_parse_ok")) for row in raw_rows
        ),
        "generated_invalid_valence_rows": sum(
            str(row.get("invalid_reason") or "")
            == "generated_invalid_valence"
            for row in raw_rows
        ),
        "generated_attribute_ambiguous_rows": sum(
            str(row.get("invalid_reason") or "")
            == "generated_attribute_ambiguous"
            for row in raw_rows
        ),
        "generated_sanitize_failure_rows": sum(
            str(row.get("invalid_reason") or "")
            == "generated_sanitize_failure"
            for row in raw_rows
        ),
    }
    recalculated["generated_invalid_other_rows"] = (
        len(raw_rows)
        - recalculated["generated_rdkit_valid_rows"]
        - recalculated["generated_invalid_valence_rows"]
    )
    recalculated["generated_valid_rate"] = (
        recalculated["generated_rdkit_valid_rows"] / len(raw_rows)
        if raw_rows
        else 0.0
    )
    for field, expected in recalculated.items():
        if summary.get(field) != expected:
            raise AssertionError(
                f"Summary mismatch for {field}: "
                f"actual={summary.get(field)!r}, expected={expected!r}"
            )
    if len(invalid) != sum(not _bool_value(row.get("rdkit_parse_ok")) for row in raw_rows):
        raise AssertionError("invalid_candidates count mismatch.")
    if len(non_target) != sum(
        _bool_value(row.get("rdkit_parse_ok"))
        and not _bool_value(row.get("teacher_target_ok"))
        for row in raw_rows
    ):
        raise AssertionError("non_target_candidates count mismatch.")
    for payload, name in ((summary, "summary"), (manifest, "manifest")):
        if payload.get("calibration_loaded", payload.get("calibration_used")) is not False:
            raise AssertionError(f"{name} does not prove calibration was unused.")
        if payload.get("test_loaded", payload.get("test_used")) is not False:
            raise AssertionError(f"{name} does not prove test was unused.")
    if manifest.get("generation_input_split") != "train":
        raise AssertionError("Manifest generation_input_split is not train.")
    if manifest.get("candidate_selection_performed") is not False:
        raise AssertionError("Manifest incorrectly reports candidate selection.")
    if manifest.get("teacher_used_only_for_target_validation") is not True:
        raise AssertionError("Manifest teacher role is incorrect.")
    if manifest_inputs.get("train_cohort_hash") not in {
        None,
        train_cohort_hash(parents),
    }:
        raise AssertionError("Manifest full train cohort hash mismatch.")
    if manifest_inputs.get("selected_train_cohort_hash") not in {
        None,
        rebuilt_selected_hash,
    }:
        raise AssertionError("Manifest selected train cohort hash mismatch.")
    if summary.get("source_codec_passed") is not True:
        raise AssertionError(
            "Summary does not prove that the source graph codec round-trip "
            "passed before native training."
        )
    checked_rows = int(summary.get("source_codec_checked_rows") or 0)
    if checked_rows != len(selected_parents):
        raise AssertionError(
            "Source codec checked-row count does not match selected parents: "
            f"checked={checked_rows}, selected={len(selected_parents)}."
        )
    if int(summary.get("source_codec_rdkit_valid_rows") or 0) != checked_rows:
        raise AssertionError("Source codec did not sanitize every selected parent.")
    if int(summary.get("source_codec_structure_match_rows") or 0) != checked_rows:
        raise AssertionError(
            "Source codec did not structurally round-trip every selected parent."
        )
    if summary.get("formal_charge_encoded_by_native") is not False:
        raise AssertionError(
            "Mutagenicity summary incorrectly reports native formal-charge "
            "encoding."
        )
    if summary.get("atom_attribute_source") != "source_anchored":
        raise AssertionError("Unexpected atom attribute source.")
    if summary.get("source_atom_mapping_unique") is not True:
        raise AssertionError("Source atom mapping is not uniquely established.")
    if int(summary.get("source_codec_invalid_valence_rows") or 0) != 0:
        raise AssertionError("Source codec has invalid-valence rows.")
    if int(
        summary.get("source_codec_attribute_mapping_failed_rows") or 0
    ) != 0:
        raise AssertionError("Source codec has atom mapping failures.")
    if require_complete and (
        summary.get("run_complete") is not True
        or manifest.get("run_complete") is not True
        or complete.get("run_complete") is not True
    ):
        raise AssertionError("GlobalGCE train pool is not complete.")
    return {
        "audit_passed": True,
        "input_train_rows": len(parents),
        "expected_input_train_rows": (
            int(expected_input_train_count)
            if expected_input_train_count is not None
            else None
        ),
        "selected_train_rows": len(selected_parents),
        "expected_selected_parent_rows": int(expected_parent_count),
        "selected_cohort_hash_matches": selected_hash_matches,
        "candidate_source_parent_rows": len(unique_source_parents),
        "candidate_source_parent_subset_of_selected": (
            candidate_sources_subset
        ),
        "source_parent_coverage_recomputed": (
            source_parent_coverage_recomputed
        ),
        "parent_limit": parent_limit,
        "selection_seed": selection_seed,
        "deterministic_parent_selection_method": selection_method,
        "raw_generated_rows": len(raw_rows),
        "candidate_pool_rows": len(pool_rows),
        "canonical_unique_candidates": len(universe),
        "train_cohort_hash": train_cohort_hash(parents),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }


__all__ = [
    "DEFAULT_EXPECTED_PARENT_COUNT",
    "DEFAULT_NATIVE_TRAIN_CSV",
    "GlobalGCECodecMetadata",
    "GlobalGCEEmptyCandidateUniverseError",
    "GlobalGCEGraphDecodeResult",
    "GlobalGCEMutagenicityCodecError",
    "NativeGenerationResult",
    "OFFICIAL_MUTAGENICITY_EDGE_LABEL_TO_BOND",
    "OFFICIAL_MUTAGENICITY_NODE_LABEL_TO_SYMBOL",
    "OfficialGlobalGCEMutagenicityGenerator",
    "PoolBuildConfig",
    "TARGET_LABEL",
    "TrainParent",
    "attach_globalgce_generation_dataset",
    "audit_mutagenicity_train_pool",
    "build_mutagenicity_train_pool",
    "decode_globalgce_molecule",
    "globalgce_tensors_to_graph_record",
    "load_strict_train_parents",
    "log_globalgce_phase_memory",
    "probe_source_graph_codec",
    "require_source_codec_gate",
    "stable_candidate_id",
    "train_cohort_hash",
    "validate_globalgce_generation_loader",
]
