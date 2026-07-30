"""Train-only GlobalGCE adaptation for the strict Mutagenicity protocol.

The project-owned layer controls cohort membership, provenance, chemistry
conversion, RF target validation, resumability, and audit. The actual rule
mining, rule optimization, and rule-to-counterfactual application are delegated
to the official GlobalGCE implementation.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

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
    "_RUN_COMPLETE.json",
)


class GlobalGCEMutagenicityCodecError(RuntimeError):
    """Raised when the native source graph codec fails its round-trip gate."""


class GlobalGCEEmptyCandidateUniverseError(RuntimeError):
    """Raised when native generation yields no valid target candidates."""


@dataclass(frozen=True, slots=True)
class GlobalGCECodecMetadata:
    """Internal label metadata used by the current native GlobalGCE tensors."""

    atom_symbols: tuple[str, ...]
    bond_names: tuple[str, ...]

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
        return cls(atom_symbols=atom_symbols, bond_names=bond_names)

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
    ) -> None:
        self.dataset_name = DATASET_NAME
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
        self.num_classes = 2
        self.index = torch_module.arange(len(parent_ids), dtype=torch_module.long)
        self.atom_symbols = list(atom_symbols)
        self.bond_names = list(bond_names)

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


def _atomic_write_text(path: Path, text: str) -> None:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
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


def stable_candidate_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(str(canonical_smiles).encode("utf-8")).hexdigest()
    return f"MUT_GLOBALGCE_{digest[:20].upper()}"


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
    required = {"molecule_id", "label"}
    smiles_col = "smiles" if rows and "smiles" in rows[0] else "parent_smiles"
    required.add(smiles_col)
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Strict train CSV is missing columns: {missing}")
    parents: list[TrainParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        split = str(row.get("split") or "train").strip().lower()
        try:
            label = int(float(str(row.get("label") or "")))
        except ValueError as exc:
            raise ValueError(f"Invalid label at {path}:{row_number}") from exc
        if not parent_id or parent_id in seen:
            raise ValueError(f"Missing/duplicate molecule_id at {path}:{row_number}")
        if split != "train":
            raise ValueError(
                f"Strict GlobalGCE input must be train-only, found split={split!r}."
            )
        if label != SOURCE_LABEL:
            raise ValueError(
                f"Strict GlobalGCE source label must be {SOURCE_LABEL}."
            )
        teacher_pred = row.get("teacher_pred")
        if teacher_pred not in (None, "") and int(float(str(teacher_pred))) != 1:
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


def _load_general_train_rows(path: Path) -> list[TrainParent]:
    _reject_non_train_path(path, description="Native GNN train CSV")
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"Native GNN train CSV is empty: {path}")
    smiles_col = "smiles" if "smiles" in rows[0] else "parent_smiles"
    parents: list[TrainParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        split = str(row.get("split") or "train").strip().lower()
        if not parent_id or parent_id in seen:
            raise ValueError(f"Invalid native train molecule_id at {path}:{row_number}")
        if split != "train":
            raise ValueError("Native GNN training data contains non-train split.")
        label = int(float(str(row.get("label") or "")))
        if label not in (0, 1):
            raise ValueError("Native GNN training labels must be binary.")
        canonical = _canonical_smiles(str(row.get(smiles_col) or ""))
        if canonical is None:
            raise ValueError(f"Invalid native train SMILES at {path}:{row_number}")
        seen.add(parent_id)
        parents.append(TrainParent(parent_id, canonical, label, "train"))
    if {parent.label for parent in parents} != {0, 1}:
        raise ValueError(
            "Official GlobalGCE requires a two-class native GNN train set. "
            "The current native train CSV does not contain both labels."
        )
    parents.sort(key=lambda row: row.parent_id)
    return parents


def _resolve_official_src(official_root: str | Path) -> Path:
    root = Path(official_root).expanduser().resolve()
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


def _build_dense_dataset(
    parents: Sequence[TrainParent],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    torch_module: Any,
    atom_symbols: Sequence[str] | None = None,
    max_num_nodes: int | None = None,
) -> _DenseMoleculeDataset:
    molecules = [_kekulized_molecule(parent.smiles) for parent in parents]
    symbols = list(atom_symbols or ())
    if not symbols:
        present_symbols = {
            atom.GetSymbol()
            for molecule in molecules
            for atom in molecule.GetAtoms()
        }
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
        labels[graph_index] = 0 if parent.label == SOURCE_LABEL else 1
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
    )


def _import_official_modules(official_src: Path) -> dict[str, Any]:
    source_text = str(official_src)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {
        "GTGNN": importlib.import_module("models.GTGNN").GTGNN,
        "GlobalGCE": importlib.import_module("models.GlobalGCE").GlobalGCE,
        "generate_cfs": importlib.import_module("models.GlobalGCE").generate_cfs,
        "train_globalgce": importlib.import_module(
            "models.models_utils"
        ).train_globalgce,
    }


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


def decode_globalgce_molecule(
    feature: Any,
    adjacency: Any,
    edge_attr: Any,
    *,
    metadata: GlobalGCECodecMetadata,
    num_nodes: int | None = None,
) -> GlobalGCEGraphDecodeResult:
    """Decode one native graph using its explicit node/edge metadata."""

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
            False, None, None, False, "codec_tensor_error", str(exc), 0, 0, {}
        )
    if record["asymmetric_adjacency_pairs"]:
        return GlobalGCEGraphDecodeResult(
            False,
            None,
            None,
            False,
            "asymmetric_adjacency",
            f"pairs={record['asymmetric_adjacency_pairs'][:5]}",
            0,
            0,
            record,
        )
    active_nodes = list(record["active_node_indices"])
    if not active_nodes:
        return GlobalGCEGraphDecodeResult(
            False, None, None, False, "empty_graph", "No real nodes.", 0, 0, record
        )
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
            old_to_new[old_index] = editable.AddAtom(Chem.Atom(symbol))
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
            "codec_graph_error",
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
        error_type = (
            "native_generated_invalid_valence"
            if "valence" in error_message.lower()
            else "native_generated_invalid_sanitize"
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
    failures: list[dict[str, Any]] = []
    for index, parent in enumerate(parents):
        graph = dataset[index]
        result = decode_globalgce_molecule(
            graph["feature"],
            graph["adj"],
            graph.get("edge_attr"),
            metadata=metadata,
            num_nodes=int(_to_builtin(graph["num_nodes"])),
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
        if not result.ok or not structure_match:
            failures.append(
                {
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "decoded_smiles": result.smiles,
                    "codec_ok": result.ok,
                    "structure_match": structure_match,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                }
            )
    checked = len(parents)
    summary = {
        "source_codec_checked_rows": checked,
        "source_codec_rdkit_valid_rows": valid_count,
        "source_codec_structure_match_rows": structure_count,
        "source_codec_passed": (
            checked > 0
            and valid_count == checked
            and structure_count == checked
        ),
        "source_codec_failure_examples": failures[:10],
        "codec_metadata": metadata.to_dict(),
    }
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


def _prepare_native_and_source_datasets(
    *,
    native_train_csv: Path,
    parents: Sequence[TrainParent],
    seed: int,
    torch_module: Any,
) -> tuple[
    list[TrainParent],
    list[int],
    list[int],
    _DenseMoleculeDataset,
    list[int],
    list[int],
    _DenseMoleculeDataset,
]:
    native_parents = _load_general_train_rows(native_train_csv)
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
    ) -> None:
        self.official_src = _resolve_official_src(official_root)
        repo_root = Path(__file__).resolve().parents[2]
        configured = native_train_csv or os.environ.get(
            "GLOBALGCE_MUTAGENICITY_NATIVE_TRAIN_CSV"
        )
        self.native_train_csv = Path(
            configured or repo_root / DEFAULT_NATIVE_TRAIN_CSV
        ).expanduser().resolve()

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
            "native_train_csv": _file_identity(self.native_train_csv),
            "official_src": str(self.official_src),
            "official_source_files": source_files,
        }

    def probe_codec(
        self,
        parents: Sequence[TrainParent],
        *,
        seed: int,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if not self.native_train_csv.is_file():
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
        )
        summary = probe_source_graph_codec(source_dataset, parents)
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
            _write_json(Path(output_path).expanduser().resolve(), summary)
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
    ) -> NativeGenerationResult:
        del resume
        if not self.native_train_csv.is_file():
            raise FileNotFoundError(
                "Official GlobalGCE requires the current two-class processed train "
                f"CSV for its native GNN: {self.native_train_csv}"
            )
        modules = _import_official_modules(self.official_src)
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
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        codec_summary = probe_source_graph_codec(source_dataset, parents)
        _write_json(output_dir / "source_codec_summary.json", codec_summary)
        require_source_codec_gate(codec_summary)
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
        gnn_checkpoint = output_dir / "native_gnn.pt"
        gnn_model = modules["GTGNN"](
            native_dataset.node_feat_dim,
            32,
            2,
            native_dataset.edge_attr_dim,
            resolved_device,
            str(gnn_checkpoint),
        ).to(resolved_device)
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

        source_prediction_loader = DataLoader(
            source_dataset,
            batch_size=500,
            shuffle=False,
        )
        native_source_indices: set[int] = set()
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
            index for index in source_train_idx if index in native_source_indices
        ]
        source_val_idx = [
            index for index in source_val_idx if index in native_source_indices
        ]
        if not source_train_idx or not source_val_idx:
            raise RuntimeError(
                "Native GlobalGCE GNN did not retain both internal train and "
                "validation source-label partitions."
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
        model_checkpoint = output_dir / "globalgce_model.pt"
        rules_checkpoint = output_dir / "globalgce_rules.pt"
        frequent_subgraphs = output_dir / "frequent_subgraphs.pkl"
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
        augmented_test_loader = modules["train_globalgce"](
            int(epochs),
            gnn_model,
            globalgce_model,
            float(learning_rate),
            source_train_loader,
            source_val_loader,
            str(rules_checkpoint),
            str(model_checkpoint),
        )
        rules = torch.load(rules_checkpoint, map_location=resolved_device)
        augmented_dataset = augmented_test_loader.dataset.dataset
        attach_globalgce_generation_dataset(
            augmented_dataset,
            source_dataset,
        )
        source_expansion_order = (
            list(source_dataset.train_idx)
            + list(source_dataset.val_idx)
            + list(source_dataset.test_idx)
        )
        all_augmented_loader = DataLoader(
            augmented_dataset,
            batch_size=500,
            shuffle=False,
        )
        validate_globalgce_generation_loader(all_augmented_loader)
        cf_feat, cf_adj, cf_edge, graph_idx = modules["generate_cfs"](
            all_augmented_loader,
            rules,
            resolved_device,
        )
        ranks: dict[str, int] = defaultdict(int)
        records: list[dict[str, Any]] = []
        codec_metadata = GlobalGCECodecMetadata.from_dataset(source_dataset)
        native_run_id = (
            f"mutagenicity_seed{seed}_epochs{epochs}_topk{top_k_native}"
        )
        for index in range(len(cf_feat)):
            source_position = int(graph_idx[index])
            if source_position < 0 or source_position >= len(source_expansion_order):
                raise RuntimeError(
                    "Official GlobalGCE returned invalid augmented source "
                    f"position {source_position}."
                )
            source_index = source_expansion_order[source_position]
            parent = parents[source_index]
            ranks[parent.parent_id] += 1
            conversion = decode_globalgce_molecule(
                cf_feat[index],
                cf_adj[index],
                cf_edge[index] if cf_edge is not None else None,
                metadata=codec_metadata,
            )
            records.append(
                {
                    "raw_smiles": conversion.smiles,
                    "source_parent_id": parent.parent_id,
                    "source_parent_smiles": parent.smiles,
                    "source_split": "train",
                    "generator_method": GENERATOR_METHOD,
                    "generator_rank": ranks[parent.parent_id],
                    "generator_score": -float(ranks[parent.parent_id]),
                    "native_rule_id": f"official_rule_application_{ranks[parent.parent_id]}",
                    "native_run_id": native_run_id,
                    "native_conversion_ok": bool(conversion.ok),
                    "native_codec_decoded": bool(conversion.codec_decoded),
                    "native_conversion_error_type": conversion.error_type,
                    "native_conversion_error": conversion.error_message,
                    "native_graph_record": conversion.graph_record,
                }
            )
        training_summary = {
            "official_entrypoints": [
                "data.data_preprocess-compatible dense molecular tensors",
                "models.GTGNN.GTGNN",
                "models.GlobalGCE.GlobalGCE",
                "models.models_utils.train_globalgce",
                "models.GlobalGCE.generate_cfs",
                "concate_inputs_with_local_recourse",
            ],
            "native_gnn_required": True,
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
            "raw_generated_rows": len(records),
            "native_source_parent_count": len(native_source_indices),
            **codec_summary,
            "saved_results_candidates_used": False,
            "generation_input_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
        }
        return NativeGenerationResult(records, training_summary)


def _stratified_native_split(
    parents: Sequence[TrainParent],
    *,
    seed: int,
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    for label in (0, 1):
        indices = [index for index, parent in enumerate(parents) if parent.label == label]
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
    for raw_index, source in enumerate(native_records):
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
                stable_candidate_id(canonical) if canonical is not None else None
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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        grouped[str(row["canonical_smiles"])].append(row)
    universe: list[dict[str, Any]] = []
    for canonical in sorted(grouped):
        occurrences = grouped[canonical]
        source_ids = sorted({str(row["source_parent_id"]) for row in occurrences})
        representative = occurrences[0]
        universe.append(
            {
                "candidate_id": stable_candidate_id(canonical),
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
    return raw_rows, pool_rows, universe, invalid_rows, non_target_rows


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
) -> dict[str, Any]:
    resolved = config or PoolBuildConfig()
    if int(resolved.epochs) <= 0 or int(resolved.top_k_native) <= 0:
        raise ValueError("epochs and top_k_native must be positive.")
    if not 0.0 <= float(resolved.dropout) < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if float(resolved.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive.")
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
            "dataset": DATASET_NAME,
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
    training_path = destination / "training_summary.json"
    checkpoint = _read_json(checkpoint_path)
    if (
        checkpoint.get("stage") in {"generated", "complete"}
        and raw_path.is_file()
        and training_path.is_file()
    ):
        native_records = _read_jsonl(raw_path)
        training_summary = _read_json(training_path)
    else:
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
        )
        native_records = list(result.records)
        training_summary = dict(result.training_summary)
        # Persist unannotated native output first; it is replaced by the full raw audit below.
        _write_jsonl(raw_path, native_records)
        _write_json(training_path, training_summary)
        _write_json(
            checkpoint_path,
            {
                "config_fingerprint": fingerprint,
                "stage": "generated",
                "raw_generated_rows": len(native_records),
                "run_complete": False,
                "updated_at": _utc_now(),
            },
        )

    raw_rows, pool_rows, universe, invalid_rows, non_target_rows = (
        _annotate_and_filter_candidates(
            native_records,
            parents=selected_parents,
            teacher=teacher,
            seed=int(resolved.seed),
        )
    )
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(destination / "candidate_pool.jsonl", pool_rows)
    _write_jsonl(destination / "candidate_universe.jsonl", universe)
    _write_jsonl(destination / "invalid_candidates.jsonl", invalid_rows)
    _write_jsonl(destination / "non_target_candidates.jsonl", non_target_rows)
    source_parent_ids = {str(row["source_parent_id"]) for row in pool_rows}
    generated_valid_rows = sum(
        _bool_value(row["rdkit_parse_ok"]) for row in raw_rows
    )
    generated_decoded_rows = sum(
        _bool_value(row.get("native_codec_decoded")) for row in raw_rows
    )
    generated_invalid_valence_rows = sum(
        str(row.get("invalid_reason") or "")
        == "native_generated_invalid_valence"
        for row in raw_rows
    )
    generated_invalid_other_rows = (
        len(raw_rows)
        - generated_valid_rows
        - generated_invalid_valence_rows
    )
    source_codec_passed = training_summary.get("source_codec_passed")
    pool_nonempty = len(pool_rows) > 0 and len(universe) > 0
    summary = {
        "input_train_rows": len(all_parents),
        "selected_train_rows": len(selected_parents),
        "unique_source_parents": len(source_parent_ids),
        "raw_generated_rows": len(raw_rows),
        "rdkit_valid_rows": generated_valid_rows,
        "teacher_target_rows": sum(
            _bool_value(row["teacher_target_ok"]) for row in raw_rows
        ),
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
        "source_codec_passed": source_codec_passed,
        "source_codec_failure_examples": list(
            training_summary.get("source_codec_failure_examples") or []
        ),
        "generated_graph_rows": len(raw_rows),
        "generated_codec_decoded_rows": generated_decoded_rows,
        "generated_rdkit_valid_rows": generated_valid_rows,
        "generated_invalid_valence_rows": generated_invalid_valence_rows,
        "generated_invalid_other_rows": generated_invalid_other_rows,
        "generated_valid_rate": (
            generated_valid_rows / len(raw_rows) if raw_rows else 0.0
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": pool_nonempty,
    }
    _write_json(destination / "summary.json", summary)
    if not pool_nonempty:
        failure = {
            "error": "empty_candidate_universe",
            "source_codec_passed": source_codec_passed,
            "raw_generated_rows": len(raw_rows),
            "generated_codec_decoded_rows": generated_decoded_rows,
            "generated_rdkit_valid_rows": generated_valid_rows,
            "generated_invalid_valence_rows": generated_invalid_valence_rows,
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
            f"raw_generated_rows={len(raw_rows)}, "
            f"generated_codec_decoded_rows={generated_decoded_rows}, "
            f"generated_rdkit_valid_rows={generated_valid_rows}, "
            f"generated_invalid_valence_rows={generated_invalid_valence_rows}, "
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
            "raw_generated_rows": len(raw_rows),
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
    require_target_label_zero: bool = True,
    require_unique_universe: bool = True,
    forbid_calibration_test: bool = True,
    require_complete: bool = True,
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
    parents, _ = load_strict_train_parents(
        train_csv,
        expected_parent_count=int(expected_parent_count),
        forbid_calibration_test=bool(forbid_calibration_test),
    )
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "run_manifest.json")
    parent_limit = int(
        (manifest.get("inputs") or {}).get("parent_limit")
        or 0
    )
    selected_parents = parents[:parent_limit] if parent_limit else list(parents)
    parent_ids = {parent.parent_id for parent in selected_parents}
    raw_rows = _read_jsonl(root / "raw_generated_candidates.jsonl")
    pool_rows = _read_jsonl(root / "candidate_pool.jsonl")
    universe = _read_jsonl(root / "candidate_universe.jsonl")
    invalid = _read_jsonl(root / "invalid_candidates.jsonl")
    non_target = _read_jsonl(root / "non_target_candidates.jsonl")
    for row in raw_rows:
        if str(row.get("source_parent_id") or "") not in parent_ids:
            raise AssertionError("Raw candidate references a non-train parent.")
        if str(row.get("source_split") or "").lower() != "train":
            raise AssertionError("Raw candidate source_split is not train.")
    for row in pool_rows:
        if str(row.get("source_parent_id") or "") not in parent_ids:
            raise AssertionError("Candidate pool references a non-train parent.")
        if str(row.get("source_split") or "").lower() != "train":
            raise AssertionError("Candidate pool source_split is not train.")
        canonical = str(row.get("canonical_smiles") or "")
        if stable_candidate_id(canonical) != row.get("candidate_id"):
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
        if stable_candidate_id(canonical) != row.get("candidate_id"):
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
            len(unique_source_parents) / len(selected_parents)
            if selected_parents
            else 0.0
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
            == "native_generated_invalid_valence"
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
    if require_complete and (
        summary.get("run_complete") is not True
        or manifest.get("run_complete") is not True
        or complete.get("run_complete") is not True
    ):
        raise AssertionError("GlobalGCE train pool is not complete.")
    return {
        "audit_passed": True,
        "input_train_rows": len(parents),
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
    "probe_source_graph_codec",
    "require_source_codec_gate",
    "stable_candidate_id",
    "train_cohort_hash",
    "validate_globalgce_generation_loader",
]
