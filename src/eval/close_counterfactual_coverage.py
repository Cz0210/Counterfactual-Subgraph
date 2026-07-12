"""Close counterfactual coverage evaluation utilities.

This module is evaluation-only. It compares selected fragment sets and
full-graph GCF-style counterfactual candidates under the same teacher and
distance-threshold protocol without changing training, reward, or selector
logic.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.eval.flip_semantics import teacher_flip_audit_fields, teacher_strict_flip
from src.rewards.teacher_semantic import TeacherSemanticScorer
from src.utils.io import ensure_directory

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on runtime
    Chem = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional fallback
    nx = None


OURS_FRAGMENT_FIELDS = (
    "final_fragment",
    "core_fragment",
    "fragment",
    "selected_fragment",
    "smiles",
    "subgraph_smiles",
)
GCF_SMILES_FIELDS = (
    "counterfactual_smiles",
    "cf_smiles",
    "graph_smiles",
    "smiles",
    "final_smiles",
    "candidate_smiles",
)
DETAIL_FIELDS = [
    "method",
    "distance_type",
    "ged_mode",
    "parent_id",
    "parent_smiles",
    "label",
    "candidate_id",
    "candidate_smiles",
    "fragment_smiles",
    "match",
    "match_index",
    "match_atoms",
    "residual_smiles",
    "delete_valid",
    "num_components",
    "num_match_atoms",
    "num_removed_atoms",
    "num_removed_bonds",
    "residual_atom_count",
    "residual_bond_count",
    "p_before",
    "p_after",
    "pred_before",
    "pred_after",
    "cf_drop",
    "cf_flip",
    "teacher_strict_flip",
    "old_weak_flip",
    "flip_definition",
    "distance",
    "cosine_similarity",
    "embedding_ok",
    "ged_ok",
    "atom_delete_ratio",
    "bond_delete_ratio",
    "error",
]
SUMMARY_FIELDS = [
    "method",
    "distance_type",
    "ged_mode",
    "threshold",
    "threshold_similarity_equivalent",
    "num_parents",
    "num_candidates",
    "num_matched_parents",
    "match_rate",
    "num_delete_valid_parents",
    "delete_valid_rate",
    "num_close_only_covered",
    "close_only_coverage",
    "num_close_cf_covered",
    "close_cf_coverage",
    "avg_best_distance",
    "median_best_distance",
    "avg_cf_drop_among_covered",
    "flip_rate_among_covered",
    "avg_atom_delete_ratio_among_covered",
    "avg_bond_delete_ratio_among_covered",
    "cache_hit_rate",
    "embedding_ok_rate",
    "ged_ok_rate",
    "total_pairs",
    "total_detail_rows",
]


@dataclass(frozen=True, slots=True)
class ParentRecord:
    parent_id: str
    smiles: str
    label: int
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    smiles: str
    raw: dict[str, Any]


class DistanceCache:
    """Small JSONL-backed cache for distance calculations."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.values: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    key = payload.get("key")
                    value = payload.get("value")
                    if isinstance(key, str) and isinstance(value, dict):
                        self.values[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self.values:
            self.hits += 1
            return dict(self.values[key])
        self.misses += 1
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.values[key] = dict(value)

    def flush(self) -> None:
        if self.path is None:
            return
        ensure_directory(self.path.parent)
        with self.path.open("w", encoding="utf-8") as handle:
            for key in sorted(self.values):
                handle.write(json.dumps({"key": key, "value": self.values[key]}, ensure_ascii=False))
                handle.write("\n")


def canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if parsing/sanitization fails."""

    if Chem is None:
        return None
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    try:
        mol = Chem.MolFromSmiles(normalized, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_from_smiles(smiles: str) -> Any | None:
    """Parse a SMILES string into a sanitized RDKit Mol."""

    if Chem is None:
        return None
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    try:
        mol = Chem.MolFromSmiles(normalized, sanitize=True)
    except Exception:
        return None
    return mol


def _clear_broken_aromatic_flags(mol: Any) -> None:
    if Chem is None or mol is None:
        return
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)


def _strip_dummy_atoms_from_query(smiles: str) -> tuple[Any | None, str | None]:
    if Chem is None:
        return None, "rdkit_unavailable"
    try:
        mol = Chem.MolFromSmiles(str(smiles or "").strip(), sanitize=False)
    except Exception as exc:
        return None, f"fragment_parse_failed:{exc}"
    if mol is None:
        return None, "fragment_parse_failed"

    editable = Chem.RWMol(mol)
    dummy_indices = sorted(
        (atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for atom_index in dummy_indices:
        editable.RemoveAtom(int(atom_index))
    query = editable.GetMol()
    if query.GetNumAtoms() == 0:
        return None, "fragment_empty_after_dummy_removal"
    try:
        Chem.SanitizeMol(query)
    except Exception:
        _clear_broken_aromatic_flags(query)
        try:
            Chem.SanitizeMol(query)
        except Exception as exc:
            return None, f"fragment_sanitize_failed:{exc}"
    return query, None


def _component_count(mol: Any) -> int | None:
    if Chem is None or mol is None:
        return None
    try:
        return len(Chem.GetMolFrags(mol))
    except Exception:
        return None


def _keep_largest_component(mol: Any) -> Any:
    if Chem is None:
        return mol
    components = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not components:
        return mol
    return max(components, key=lambda item: (item.GetNumAtoms(), item.GetNumBonds()))


def _incident_bond_count(parent_mol: Any, atom_indices: set[int]) -> int:
    return sum(
        1
        for bond in parent_mol.GetBonds()
        if int(bond.GetBeginAtomIdx()) in atom_indices or int(bond.GetEndAtomIdx()) in atom_indices
    )


def hard_delete_substructure_any_match(
    parent_smiles: str,
    fragment_smiles: str,
    sanitize: bool = True,
    keep_components: str = "all",
) -> list[dict[str, Any]]:
    """Delete every RDKit substructure match and return all residual candidates."""

    if keep_components not in {"all", "largest"}:
        raise ValueError("keep_components must be one of {'all', 'largest'}")
    if Chem is None:
        return []

    parent_mol = mol_from_smiles(parent_smiles)
    if parent_mol is None:
        return []
    query, query_error = _strip_dummy_atoms_from_query(fragment_smiles)
    if query is None:
        return []

    try:
        matches = parent_mol.GetSubstructMatches(query, useChirality=True, uniquify=True)
    except Exception:
        return []

    candidates: list[dict[str, Any]] = []
    parent_atom_count = int(parent_mol.GetNumAtoms())
    parent_bond_count = int(parent_mol.GetNumBonds())
    for match_index, match in enumerate(matches):
        match_atoms = [int(index) for index in match]
        match_atom_set = set(match_atoms)
        removed_bonds = _incident_bond_count(parent_mol, match_atom_set)
        payload: dict[str, Any] = {
            "match_index": int(match_index),
            "match_atoms": match_atoms,
            "num_match_atoms": len(match_atoms),
            "num_removed_atoms": len(match_atoms),
            "num_removed_bonds": int(removed_bonds),
            "residual_smiles": None,
            "delete_valid": False,
            "num_components": None,
            "residual_atom_count": None,
            "residual_bond_count": None,
            "atom_delete_ratio": (len(match_atoms) / parent_atom_count) if parent_atom_count else None,
            "bond_delete_ratio": (removed_bonds / parent_bond_count) if parent_bond_count else 0.0,
            "error": query_error,
        }
        try:
            editable = Chem.RWMol(parent_mol)
            for atom_index in sorted(match_atoms, reverse=True):
                editable.RemoveAtom(int(atom_index))
            residual_mol = editable.GetMol()
            if residual_mol.GetNumAtoms() == 0:
                payload["error"] = "empty_residual_after_deletion"
                candidates.append(payload)
                continue
            if keep_components == "largest":
                residual_mol = _keep_largest_component(residual_mol)
                if residual_mol.GetNumAtoms() == 0:
                    payload["error"] = "empty_largest_component_after_deletion"
                    candidates.append(payload)
                    continue
            if sanitize:
                try:
                    Chem.SanitizeMol(residual_mol)
                except Exception:
                    _clear_broken_aromatic_flags(residual_mol)
                    try:
                        Chem.SanitizeMol(residual_mol)
                    except Exception as exc:
                        payload["error"] = f"residual_sanitize_failed:{exc}"
                        candidates.append(payload)
                        continue
            residual_smiles = Chem.MolToSmiles(residual_mol, canonical=True)
            if not residual_smiles:
                payload["error"] = "empty_residual_smiles"
                candidates.append(payload)
                continue
            payload.update(
                {
                    "residual_smiles": residual_smiles,
                    "delete_valid": True,
                    "num_components": _component_count(residual_mol),
                    "residual_atom_count": int(residual_mol.GetNumAtoms()),
                    "residual_bond_count": int(residual_mol.GetNumBonds()),
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive around RDKit graph edits
            payload["error"] = f"delete_failed:{exc}"
        candidates.append(payload)
    return candidates


def mol_to_labeled_nx_graph(mol: Any) -> Any:
    """Convert an RDKit Mol into a labeled NetworkX graph for GED."""

    if nx is None:
        raise RuntimeError("networkx is required for graph_edit_distance mode.")
    if Chem is None or mol is None:
        raise ValueError("A valid RDKit Mol is required.")
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            int(atom.GetIdx()),
            atomic_num=int(atom.GetAtomicNum()),
            formal_charge=int(atom.GetFormalCharge()),
            is_aromatic=bool(atom.GetIsAromatic()),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            bond_type=str(bond.GetBondType()),
            is_aromatic=bool(bond.GetIsAromatic()),
        )
    return graph


def normalized_delete_ged_distance(
    parent_mol: Any,
    residual_mol: Any,
    num_removed_atoms: int,
    num_removed_bonds: int,
) -> float:
    """Fast normalized deletion-GED upper bound for hard-deletion residuals."""

    parent_atoms = int(parent_mol.GetNumAtoms())
    residual_atoms = int(residual_mol.GetNumAtoms())
    parent_bonds = int(parent_mol.GetNumBonds())
    residual_bonds = int(residual_mol.GetNumBonds())
    denominator = parent_atoms + residual_atoms + parent_bonds + residual_bonds
    if denominator <= 0:
        return 1.0
    ged_delete = max(0, int(num_removed_atoms)) + max(0, int(num_removed_bonds))
    return max(0.0, min(1.0, float(ged_delete / denominator)))


def normalized_networkx_ged_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 2.0,
) -> float | None:
    """Normalized exact/timeout-bounded NetworkX graph edit distance."""

    if nx is None:
        return None
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    graph_a = mol_to_labeled_nx_graph(mol_a)
    graph_b = mol_to_labeled_nx_graph(mol_b)

    def node_subst_cost(attrs_a: dict[str, Any], attrs_b: dict[str, Any]) -> float:
        return 0.0 if attrs_a == attrs_b else 1.0

    def edge_subst_cost(attrs_a: dict[str, Any], attrs_b: dict[str, Any]) -> float:
        return 0.0 if attrs_a == attrs_b else 1.0

    try:
        distance = nx.graph_edit_distance(
            graph_a,
            graph_b,
            node_subst_cost=node_subst_cost,
            node_del_cost=lambda attrs: 1.0,
            node_ins_cost=lambda attrs: 1.0,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=lambda attrs: 1.0,
            edge_ins_cost=lambda attrs: 1.0,
            timeout=float(timeout),
        )
    except TypeError:
        try:
            iterator = nx.optimize_graph_edit_distance(
                graph_a,
                graph_b,
                node_subst_cost=node_subst_cost,
                node_del_cost=lambda attrs: 1.0,
                node_ins_cost=lambda attrs: 1.0,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=lambda attrs: 1.0,
                edge_ins_cost=lambda attrs: 1.0,
            )
            distance = next(iterator, None)
        except Exception:
            return None
    except Exception:
        return None

    if distance is None:
        return None
    denominator = (
        int(graph_a.number_of_nodes())
        + int(graph_b.number_of_nodes())
        + int(graph_a.number_of_edges())
        + int(graph_b.number_of_edges())
    )
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, float(distance) / float(denominator)))


def _as_vector(value: Any) -> list[float]:
    if value is None:
        raise ValueError("embedding value is None")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    vector = [float(item) for item in value]
    if not vector or not all(math.isfinite(item) for item in vector):
        raise ValueError("embedding vector is empty or non-finite")
    return vector


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"embedding dimensions differ: {len(vec_a)} vs {len(vec_b)}")
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("embedding vector has zero norm")
    cosine = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, float(cosine)))


def _call_embedding_api(target: Any, smiles: str, embedding_layer: str) -> Any:
    method_names = (
        "embed_smiles",
        "encode_smiles",
        "get_graph_embedding",
        "get_graph_embeddings",
        "get_embedding",
        "get_embeddings",
    )
    for name in method_names:
        method = getattr(target, name, None)
        if method is None:
            continue
        try:
            return method(smiles, embedding_layer=embedding_layer)
        except TypeError:
            try:
                return method(smiles, layer=embedding_layer)
            except TypeError:
                try:
                    return method(smiles)
                except TypeError:
                    continue
    raise AttributeError("teacher_embedding_api_unavailable")


def embedding_distance_from_teacher(
    teacher: Any,
    smiles_a: str,
    smiles_b: str,
    embedding_layer: str = "penultimate",
) -> dict[str, Any]:
    """Return 1 - cosine similarity from a teacher graph embedding API."""

    try:
        source = teacher
        try:
            emb_a = _call_embedding_api(source, smiles_a, embedding_layer)
            emb_b = _call_embedding_api(source, smiles_b, embedding_layer)
        except AttributeError:
            source = getattr(teacher, "model", None)
            if source is None:
                raise
            emb_a = _call_embedding_api(source, smiles_a, embedding_layer)
            emb_b = _call_embedding_api(source, smiles_b, embedding_layer)
        vec_a = _as_vector(emb_a)
        vec_b = _as_vector(emb_b)
        cosine = _cosine_similarity(vec_a, vec_b)
        return {
            "cosine_similarity": float(cosine),
            "embedding_distance": float(1.0 - cosine),
            "embedding_ok": True,
            "embedding_error": None,
        }
    except Exception as exc:
        return {
            "cosine_similarity": None,
            "embedding_distance": None,
            "embedding_ok": False,
            "embedding_error": str(exc),
        }


def predict_with_teacher(teacher: Any, smiles: str, label: int) -> dict[str, Any]:
    """Predict one molecule with the existing teacher scorer interface."""

    normalized = str(smiles or "").strip()
    try:
        result = teacher.score_smiles(normalized, label=int(label))
    except Exception as exc:
        return {
            "pred_label": None,
            "p_label": None,
            "probs": [],
            "ok": False,
            "error": f"teacher_score_failed:{exc}",
        }
    if not result.get("teacher_result_ok"):
        return {
            "pred_label": None,
            "p_label": None,
            "probs": [],
            "ok": False,
            "error": str(result.get("teacher_reason") or "teacher_result_not_ok"),
        }
    p_label = result.get("teacher_prob")
    pred_label = result.get("teacher_label")
    probs: list[float] = []
    if p_label is not None:
        p = float(p_label)
        if int(label) == 0:
            probs = [p, 1.0 - p]
        elif int(label) == 1:
            probs = [1.0 - p, p]
    return {
        "pred_label": int(pred_label) if pred_label is not None else None,
        "p_label": float(p_label) if p_label is not None else None,
        "probs": probs,
        "ok": True,
        "error": None,
    }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("selected_rows", "candidates", "rows", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
        return [dict(payload)]
    raise ValueError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _load_table(path_like: str | Path, *, directory_candidates: Sequence[str] = ()) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(path_like).expanduser()
    if path.is_dir():
        for name in directory_candidates:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No supported input file found under directory: {path}")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path, _read_csv_rows(path)
    if suffix == ".jsonl":
        return path, _read_jsonl_rows(path)
    if suffix == ".json":
        return path, _read_json_rows(path)
    raise ValueError(f"Unsupported input format: {path}")


def _coalesce(row: dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _parse_int_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _resolve_label_col(rows: list[dict[str, Any]], requested: str) -> str:
    if not rows:
        return requested
    fields = set(rows[0].keys())
    if requested in fields:
        return requested
    for fallback in ("label", "HIV_active", "target", "y", "activity"):
        if fallback in fields:
            print(f"[LOAD_DATASET] requested_label_col={requested} missing; using {fallback}")
            return fallback
    raise ValueError(f"Dataset does not contain label column {requested!r}; available={sorted(fields)}")


def _load_parent_records(
    dataset_csv: str | Path,
    *,
    label: int,
    smiles_col: str,
    label_col: str,
    max_parents: int | None,
) -> tuple[Path, list[ParentRecord], str]:
    path, rows = _load_table(dataset_csv)
    actual_label_col = _resolve_label_col(rows, label_col)
    parents: list[ParentRecord] = []
    for row_index, row in enumerate(rows):
        smiles = str(row.get(smiles_col) or "").strip()
        parsed_label = _parse_int_label(row.get(actual_label_col))
        if not smiles or parsed_label is None or int(parsed_label) != int(label):
            continue
        parent_id = str(_coalesce(row, ("id", "parent_id", "index", "row_id")) or row_index)
        parents.append(ParentRecord(parent_id=parent_id, smiles=smiles, label=int(parsed_label), raw=row))
        if max_parents is not None and len(parents) >= int(max_parents):
            break
    return path, parents, actual_label_col


def _load_candidate_records(
    path_like: str | Path,
    *,
    fields: Sequence[str],
    directory_candidates: Sequence[str],
) -> tuple[Path, list[CandidateRecord]]:
    path, rows = _load_table(path_like, directory_candidates=directory_candidates)
    candidates: list[CandidateRecord] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        smiles = str(_coalesce(row, fields) or "").strip()
        if not smiles:
            continue
        candidate_id = str(_coalesce(row, ("candidate_id", "id", "rank", "candidate_index", "index")) or row_index)
        key = f"{candidate_id}\t{smiles}"
        if key in seen:
            continue
        seen.add(key)
        candidates.append(CandidateRecord(candidate_id=candidate_id, smiles=smiles, raw=row))
    return path, candidates


def _distance_cache_key(
    *,
    distance_type: str,
    ged_mode: str,
    smiles_a: str,
    smiles_b: str,
    teacher_path: str | Path | None,
    embedding_layer: str,
) -> str:
    canonical_a = canonicalize_smiles(smiles_a) or str(smiles_a or "").strip()
    canonical_b = canonicalize_smiles(smiles_b) or str(smiles_b or "").strip()
    return "|".join(
        [
            str(distance_type),
            str(ged_mode),
            canonical_a,
            canonical_b,
            str(teacher_path or ""),
            str(embedding_layer),
        ]
    )


def _compute_pair_distance(
    *,
    teacher: Any,
    teacher_path: str | Path | None,
    parent_smiles: str,
    candidate_smiles: str,
    distance_type: str,
    ged_mode: str,
    cache: DistanceCache,
    embedding_layer: str = "penultimate",
    delete_candidate: dict[str, Any] | None = None,
    networkx_timeout: float = 2.0,
) -> dict[str, Any]:
    key = _distance_cache_key(
        distance_type=distance_type,
        ged_mode=ged_mode,
        smiles_a=parent_smiles,
        smiles_b=candidate_smiles,
        teacher_path=teacher_path,
        embedding_layer=embedding_layer,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    if distance_type == "embedding":
        value = embedding_distance_from_teacher(
            teacher,
            parent_smiles,
            candidate_smiles,
            embedding_layer=embedding_layer,
        )
        result = {
            "distance": value.get("embedding_distance"),
            "cosine_similarity": value.get("cosine_similarity"),
            "embedding_ok": bool(value.get("embedding_ok")),
            "ged_ok": None,
            "error": value.get("embedding_error"),
        }
        cache.set(key, result)
        return result

    if distance_type != "ged":
        result = {
            "distance": None,
            "cosine_similarity": None,
            "embedding_ok": None,
            "ged_ok": False,
            "error": f"unsupported_distance_type:{distance_type}",
        }
        cache.set(key, result)
        return result

    if ged_mode == "delete" and delete_candidate is not None and delete_candidate.get("delete_valid"):
        parent_mol = mol_from_smiles(parent_smiles)
        residual_mol = mol_from_smiles(candidate_smiles)
        if parent_mol is not None and residual_mol is not None:
            distance = normalized_delete_ged_distance(
                parent_mol,
                residual_mol,
                int(delete_candidate.get("num_removed_atoms") or 0),
                int(delete_candidate.get("num_removed_bonds") or 0),
            )
            result = {
                "distance": distance,
                "cosine_similarity": None,
                "embedding_ok": None,
                "ged_ok": True,
                "error": None,
            }
        else:
            result = {
                "distance": None,
                "cosine_similarity": None,
                "embedding_ok": None,
                "ged_ok": False,
                "error": "delete_ged_parse_failed",
            }
        cache.set(key, result)
        return result

    distance = normalized_networkx_ged_distance(
        parent_smiles,
        candidate_smiles,
        timeout=float(networkx_timeout),
    )
    result = {
        "distance": distance,
        "cosine_similarity": None,
        "embedding_ok": None,
        "ged_ok": distance is not None,
        "error": None if distance is not None else "networkx_ged_failed_or_unavailable",
    }
    cache.set(key, result)
    return result


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ok"}


def _cf_condition(
    row: dict[str, Any],
    *,
    label: int,
    require_flip_only: bool,
    min_cf_drop: float,
    desired_label: int | None = None,
) -> bool:
    pred_after = _parse_int_label(row.get("pred_after"))
    cf_drop = _as_float(row.get("cf_drop"))
    cf_flip = teacher_strict_flip(row.get("pred_before"), pred_after, label)
    if desired_label is not None:
        return pred_after == int(desired_label)
    if require_flip_only:
        return cf_flip
    return bool(cf_flip or (cf_drop is not None and cf_drop >= float(min_cf_drop)))


def _pick_best_row(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    label: int,
    require_flip_only: bool,
    min_cf_drop: float,
    desired_label: int | None,
) -> tuple[dict[str, Any] | None, bool, bool]:
    eligible: list[tuple[bool, float, float, dict[str, Any]]] = []
    close_only = False
    close_cf = False
    for row in rows:
        distance = _as_float(row.get("distance"))
        if distance is None or distance > float(threshold):
            continue
        close_only = True
        is_cf = _cf_condition(
            row,
            label=label,
            require_flip_only=require_flip_only,
            min_cf_drop=min_cf_drop,
            desired_label=desired_label,
        )
        close_cf = close_cf or is_cf
        cf_drop = _as_float(row.get("cf_drop"))
        eligible.append((is_cf, distance, -(cf_drop if cf_drop is not None else -1e9), row))
    if not eligible:
        return None, close_only, close_cf
    eligible.sort(key=lambda item: (not item[0], item[1], item[2]))
    return eligible[0][3], close_only, close_cf


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else None


def _threshold_similarity_equivalent(distance_type: str, threshold: float) -> float | None:
    if distance_type != "embedding":
        return None
    return max(-1.0, min(1.0, float(1.0 - threshold)))


def build_threshold_summary(
    detail_rows: list[dict[str, Any]],
    *,
    method: str,
    distance_type: str,
    ged_mode: str,
    thresholds: Sequence[float],
    total_parents: int,
    total_candidates: int,
    require_flip_only: bool,
    min_cf_drop: float,
    desired_label: int | None = None,
    cache_hit_rate: float = 0.0,
) -> list[dict[str, Any]]:
    """Aggregate pair/match rows into parent-deduplicated threshold coverage."""

    rows_by_parent: dict[str, list[dict[str, Any]]] = {}
    labels_by_parent: dict[str, int] = {}
    for row in detail_rows:
        parent_id = str(row.get("parent_id") or "")
        rows_by_parent.setdefault(parent_id, []).append(row)
        parsed_label = _parse_int_label(row.get("label"))
        if parsed_label is not None:
            labels_by_parent[parent_id] = parsed_label

    matched_parents = {
        parent_id
        for parent_id, rows in rows_by_parent.items()
        if any(_as_bool(row.get("match")) for row in rows)
    }
    delete_valid_parents = {
        parent_id
        for parent_id, rows in rows_by_parent.items()
        if any(_as_bool(row.get("delete_valid")) for row in rows)
    }
    embedding_ok_count = sum(1 for row in detail_rows if _as_bool(row.get("embedding_ok")))
    ged_ok_count = sum(1 for row in detail_rows if _as_bool(row.get("ged_ok")))
    total_detail_rows = len(detail_rows)
    total_pairs = max(1, total_parents * max(1, total_candidates))

    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        close_only_parents: set[str] = set()
        close_cf_parents: set[str] = set()
        best_rows: list[dict[str, Any]] = []
        for parent_id, rows in rows_by_parent.items():
            label = labels_by_parent.get(parent_id, 0)
            best_row, close_only, close_cf = _pick_best_row(
                rows,
                threshold=float(threshold),
                label=label,
                require_flip_only=require_flip_only,
                min_cf_drop=min_cf_drop,
                desired_label=desired_label,
            )
            if close_only:
                close_only_parents.add(parent_id)
            if close_cf:
                close_cf_parents.add(parent_id)
            if best_row is not None and close_cf:
                best_rows.append(best_row)
        summaries.append(
            {
                "method": method,
                "distance_type": distance_type,
                "ged_mode": ged_mode,
                "threshold": float(threshold),
                "threshold_similarity_equivalent": _threshold_similarity_equivalent(distance_type, float(threshold)),
                "num_parents": int(total_parents),
                "num_candidates": int(total_candidates),
                "num_matched_parents": len(matched_parents),
                "match_rate": _safe_rate(len(matched_parents), total_parents),
                "num_delete_valid_parents": len(delete_valid_parents),
                "delete_valid_rate": _safe_rate(len(delete_valid_parents), total_parents),
                "num_close_only_covered": len(close_only_parents),
                "close_only_coverage": _safe_rate(len(close_only_parents), total_parents),
                "num_close_cf_covered": len(close_cf_parents),
                "close_cf_coverage": _safe_rate(len(close_cf_parents), total_parents),
                "avg_best_distance": _mean(_as_float(row.get("distance")) for row in best_rows),
                "median_best_distance": _median(_as_float(row.get("distance")) for row in best_rows),
                "avg_cf_drop_among_covered": _mean(_as_float(row.get("cf_drop")) for row in best_rows),
                "flip_rate_among_covered": _mean(1.0 if _as_bool(row.get("cf_flip")) else 0.0 for row in best_rows),
                "avg_atom_delete_ratio_among_covered": _mean(
                    _as_float(row.get("atom_delete_ratio")) for row in best_rows
                ),
                "avg_bond_delete_ratio_among_covered": _mean(
                    _as_float(row.get("bond_delete_ratio")) for row in best_rows
                ),
                "cache_hit_rate": float(cache_hit_rate),
                "embedding_ok_rate": _safe_rate(embedding_ok_count, total_detail_rows)
                if distance_type == "embedding"
                else None,
                "ged_ok_rate": _safe_rate(ged_ok_count, total_detail_rows)
                if distance_type == "ged"
                else None,
                "total_pairs": int(total_pairs),
                "total_detail_rows": int(total_detail_rows),
            }
        )
    return summaries


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    destination = Path(path).expanduser().resolve()
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser().resolve()
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_float(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def render_report(
    *,
    method_name: str,
    distance_type: str,
    ged_mode: str,
    thresholds: Sequence[float],
    summaries: list[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    match_metric: bool,
) -> str:
    lines = [
        "# Close Counterfactual Coverage Report",
        "",
        f"- method_name: {method_name}",
        f"- distance_type: {distance_type}",
        f"- ged_mode: {ged_mode}",
        f"- thresholds: {', '.join(str(float(item)) for item in thresholds)}",
        f"- total_parents: {total_parents}",
        f"- total_candidates: {total_candidates}",
        "",
    ]
    if distance_type == "embedding":
        lines.extend(
            [
                "Embedding distance uses `embedding_distance = 1 - cosine_similarity`.",
                "- distance <= 0.10 means cosine_similarity >= 0.90",
                "- distance <= 0.20 means cosine_similarity >= 0.80",
                "- distance <= 0.30 means cosine_similarity >= 0.70",
                "",
            ]
        )
    lines.extend(
        [
            "低成本翻转 coverage@0.20 可以视为 close counterfactual coverage 的一个特例。当候选反事实图定义为 hard deletion 后的 residual graph G\\s，距离函数使用 normalized GED 或 embedding distance，并固定 threshold=0.20，同时要求 teacher prediction flip，则该指标等价于 CloseCFCoverage@0.20。区别是 GCFExplainer 原始定义的候选是完整 counterfactual graph C，而 ours 的候选首先是 selected subgraph s，需要通过 hard deletion 映射为 G\\s。",
            "",
            "| threshold | close_only_coverage | close_cf_coverage | avg_best_distance | median_best_distance | avg_cf_drop_among_covered | flip_rate_among_covered | match_rate | delete_valid_rate | embedding_ok_rate | ged_ok_rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {threshold} | {close_only} | {close_cf} | {avg_dist} | {med_dist} | {cf_drop} | {flip_rate} | {match_rate} | {delete_rate} | {emb_rate} | {ged_rate} |".format(
                threshold=_format_float(row.get("threshold")),
                close_only=_format_float(row.get("close_only_coverage")),
                close_cf=_format_float(row.get("close_cf_coverage")),
                avg_dist=_format_float(row.get("avg_best_distance")),
                med_dist=_format_float(row.get("median_best_distance")),
                cf_drop=_format_float(row.get("avg_cf_drop_among_covered")),
                flip_rate=_format_float(row.get("flip_rate_among_covered")),
                match_rate=_format_float(row.get("match_rate")) if match_metric else "n/a",
                delete_rate=_format_float(row.get("delete_valid_rate")) if match_metric else "n/a",
                emb_rate=_format_float(row.get("embedding_ok_rate")),
                ged_rate=_format_float(row.get("ged_ok_rate")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: str | Path,
    details: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    report: str,
) -> dict[str, str]:
    destination = ensure_directory(Path(output_dir).expanduser().resolve())
    details_path = destination / "details.csv"
    summary_path = destination / "threshold_summary.csv"
    summary_json_path = destination / "threshold_summary.json"
    report_path = destination / "report.md"
    _write_csv(details_path, details, DETAIL_FIELDS)
    _write_csv(summary_path, summaries, SUMMARY_FIELDS)
    _write_json(summary_json_path, {"threshold_summary": summaries})
    report_path.write_text(report, encoding="utf-8")
    return {
        "details_csv": str(details_path),
        "threshold_summary_csv": str(summary_path),
        "threshold_summary_json": str(summary_json_path),
        "report_md": str(report_path),
    }


def _row_base(
    *,
    method: str,
    distance_type: str,
    ged_mode: str,
    parent: ParentRecord,
    candidate: CandidateRecord,
) -> dict[str, Any]:
    return {
        "method": method,
        "distance_type": distance_type,
        "ged_mode": ged_mode,
        "parent_id": parent.parent_id,
        "parent_smiles": parent.smiles,
        "label": parent.label,
        "candidate_id": candidate.candidate_id,
        "candidate_smiles": candidate.smiles,
        "fragment_smiles": candidate.smiles,
        "match": False,
        "match_index": "",
        "match_atoms": [],
        "residual_smiles": None,
        "delete_valid": False,
        "num_components": None,
        "num_match_atoms": None,
        "num_removed_atoms": None,
        "num_removed_bonds": None,
        "residual_atom_count": None,
        "residual_bond_count": None,
        "p_before": None,
        "p_after": None,
        "pred_before": None,
        "pred_after": None,
        "cf_drop": None,
        "cf_flip": False,
        "teacher_strict_flip": False,
        "old_weak_flip": False,
        "flip_definition": "pred_before == target_label and pred_after != target_label",
        "distance": None,
        "cosine_similarity": None,
        "embedding_ok": None,
        "ged_ok": None,
        "atom_delete_ratio": None,
        "bond_delete_ratio": None,
        "error": None,
    }


def evaluate_ours_selected_subgraphs(
    dataset_csv: str | Path,
    selected_subgraphs_path: str | Path,
    teacher_path: str | Path,
    label: int,
    distance_type: str,
    thresholds: Sequence[float],
    output_dir: str | Path,
    smiles_col: str = "smiles",
    label_col: str = "label",
    ged_mode: str = "delete",
    any_match: bool = True,
    hard_delete: bool = True,
    require_flip_only: bool = False,
    min_cf_drop: float = 0.0,
    max_parents: int | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate selected fragments after hard deletion from each parent graph."""

    if not hard_delete:
        raise ValueError("Only hard_delete=True is supported by close counterfactual coverage.")
    if not any_match:
        raise ValueError("Only any_match=True is currently supported.")
    method = "ours_selected_subgraphs"
    output_root = ensure_directory(Path(output_dir).expanduser().resolve())
    cache = DistanceCache(cache_path or output_root / "cache" / "dist_cache.jsonl")
    teacher = TeacherSemanticScorer(teacher_path)

    print(
        f"[CLOSE_CF_CONFIG] method={method} distance_type={distance_type} ged_mode={ged_mode} "
        f"thresholds={','.join(str(float(item)) for item in thresholds)} label={label}"
    )
    dataset_path, parents, actual_label_col = _load_parent_records(
        dataset_csv,
        label=int(label),
        smiles_col=smiles_col,
        label_col=label_col,
        max_parents=max_parents,
    )
    print(
        f"[LOAD_DATASET] path={dataset_path} label={label} smiles_col={smiles_col} "
        f"label_col={actual_label_col} parents={len(parents)}"
    )
    candidates_path, candidates = _load_candidate_records(
        selected_subgraphs_path,
        fields=OURS_FRAGMENT_FIELDS,
        directory_candidates=(
            "selected_subgraphs.json",
            "selected_subgraphs.csv",
            "selected_subgraphs.jsonl",
            "candidate_pool.jsonl",
            "candidate_pool.csv",
        ),
    )
    print(f"[LOAD_CANDIDATES] path={candidates_path} candidates={len(candidates)}")

    details: list[dict[str, Any]] = []
    before_cache: dict[str, dict[str, Any]] = {}
    started = time.time()
    for parent in parents:
        before = before_cache.get(parent.smiles)
        if before is None:
            before = predict_with_teacher(teacher, parent.smiles, parent.label)
            before_cache[parent.smiles] = before
        for candidate in candidates:
            base = _row_base(
                method=method,
                distance_type=distance_type,
                ged_mode=ged_mode,
                parent=parent,
                candidate=candidate,
            )
            deletion_candidates = hard_delete_substructure_any_match(parent.smiles, candidate.smiles)
            if not deletion_candidates:
                row = dict(base)
                row.update(
                    {
                        "p_before": before.get("p_label"),
                        "pred_before": before.get("pred_label"),
                        "error": "no_substructure_match_or_fragment_parse_failed",
                    }
                )
                details.append(row)
                continue
            for deletion in deletion_candidates:
                row = dict(base)
                row.update(
                    {
                        "match": True,
                        "match_index": deletion.get("match_index"),
                        "match_atoms": deletion.get("match_atoms") or [],
                        "residual_smiles": deletion.get("residual_smiles"),
                        "delete_valid": bool(deletion.get("delete_valid")),
                        "num_components": deletion.get("num_components"),
                        "num_match_atoms": deletion.get("num_match_atoms"),
                        "num_removed_atoms": deletion.get("num_removed_atoms"),
                        "num_removed_bonds": deletion.get("num_removed_bonds"),
                        "residual_atom_count": deletion.get("residual_atom_count"),
                        "residual_bond_count": deletion.get("residual_bond_count"),
                        "p_before": before.get("p_label"),
                        "pred_before": before.get("pred_label"),
                        "atom_delete_ratio": deletion.get("atom_delete_ratio"),
                        "bond_delete_ratio": deletion.get("bond_delete_ratio"),
                        "error": deletion.get("error"),
                    }
                )
                residual_smiles = str(deletion.get("residual_smiles") or "")
                if not deletion.get("delete_valid") or not residual_smiles:
                    details.append(row)
                    continue
                after = predict_with_teacher(teacher, residual_smiles, parent.label)
                row.update(
                    {
                        "p_after": after.get("p_label"),
                        "pred_after": after.get("pred_label"),
                        "cf_drop": (
                            float(before["p_label"]) - float(after["p_label"])
                            if before.get("ok") and after.get("ok")
                            else None
                        ),
                        "error": after.get("error") if not after.get("ok") else row.get("error"),
                    }
                )
                row.update(
                    teacher_flip_audit_fields(
                        before.get("pred_label"),
                        after.get("pred_label"),
                        parent.label,
                    )
                )
                distance_result = _compute_pair_distance(
                    teacher=teacher,
                    teacher_path=teacher_path,
                    parent_smiles=parent.smiles,
                    candidate_smiles=residual_smiles,
                    distance_type=distance_type,
                    ged_mode=ged_mode,
                    cache=cache,
                    delete_candidate=deletion,
                )
                row.update(
                    {
                        "distance": distance_result.get("distance"),
                        "cosine_similarity": distance_result.get("cosine_similarity"),
                        "embedding_ok": distance_result.get("embedding_ok"),
                        "ged_ok": distance_result.get("ged_ok"),
                        "error": row.get("error") or distance_result.get("error"),
                    }
                )
                details.append(row)
    cache.flush()
    summaries = build_threshold_summary(
        details,
        method=method,
        distance_type=distance_type,
        ged_mode=ged_mode,
        thresholds=thresholds,
        total_parents=len(parents),
        total_candidates=len(candidates),
        require_flip_only=require_flip_only,
        min_cf_drop=min_cf_drop,
        cache_hit_rate=cache.hit_rate,
    )
    report = render_report(
        method_name=method,
        distance_type=distance_type,
        ged_mode=ged_mode,
        thresholds=thresholds,
        summaries=summaries,
        total_parents=len(parents),
        total_candidates=len(candidates),
        match_metric=True,
    )
    outputs = _write_outputs(output_dir=output_root, details=details, summaries=summaries, report=report)
    print(
        f"[EVAL_OURS] parents={len(parents)} candidates={len(candidates)} detail_rows={len(details)} "
        f"seconds={time.time() - started:.2f}"
    )
    print(f"[DISTANCE_CACHE] path={cache.path} hits={cache.hits} misses={cache.misses} hit_rate={cache.hit_rate:.4f}")
    if summaries:
        last = summaries[-1]
        print(
            f"[THRESHOLD_SUMMARY] threshold={last['threshold']} close_only={last['close_only_coverage']:.4f} "
            f"close_cf={last['close_cf_coverage']:.4f}"
        )
    return {"details": details, "threshold_summary": summaries, "outputs": outputs}


def evaluate_gcf_counterfactual_graphs(
    dataset_csv: str | Path,
    gcf_candidates_path: str | Path,
    teacher_path: str | Path,
    label: int,
    distance_type: str,
    thresholds: Sequence[float],
    output_dir: str | Path,
    smiles_col: str = "smiles",
    label_col: str = "label",
    desired_label: int | None = None,
    require_flip_only: bool = True,
    min_cf_drop: float = 0.0,
    max_parents: int | None = None,
    cache_path: str | Path | None = None,
    ged_mode: str = "networkx",
) -> dict[str, Any]:
    """Evaluate full-graph GCF baseline candidates without hard deletion."""

    method = "gcf_counterfactual_graphs"
    effective_ged_mode = "networkx" if distance_type == "ged" else ged_mode
    output_root = ensure_directory(Path(output_dir).expanduser().resolve())
    cache = DistanceCache(cache_path or output_root / "cache" / "dist_cache.jsonl")
    teacher = TeacherSemanticScorer(teacher_path)
    print(
        f"[CLOSE_CF_CONFIG] method={method} distance_type={distance_type} ged_mode={effective_ged_mode} "
        f"thresholds={','.join(str(float(item)) for item in thresholds)} label={label}"
    )
    dataset_path, parents, actual_label_col = _load_parent_records(
        dataset_csv,
        label=int(label),
        smiles_col=smiles_col,
        label_col=label_col,
        max_parents=max_parents,
    )
    print(
        f"[LOAD_DATASET] path={dataset_path} label={label} smiles_col={smiles_col} "
        f"label_col={actual_label_col} parents={len(parents)}"
    )
    candidates_path, candidates = _load_candidate_records(
        gcf_candidates_path,
        fields=GCF_SMILES_FIELDS,
        directory_candidates=(
            "gcf_candidates.jsonl",
            "gcf_candidates.csv",
            "gt_selected_fullgraphs.csv",
            "camc_gt_fullgraph_selected_motifs.csv",
            "counterfactual_graphs.jsonl",
            "counterfactual_graphs.csv",
        ),
    )
    print(f"[LOAD_CANDIDATES] path={candidates_path} candidates={len(candidates)}")

    details: list[dict[str, Any]] = []
    before_cache: dict[str, dict[str, Any]] = {}
    candidate_pred_cache: dict[str, dict[str, Any]] = {}
    started = time.time()
    for parent in parents:
        before = before_cache.get(parent.smiles)
        if before is None:
            before = predict_with_teacher(teacher, parent.smiles, parent.label)
            before_cache[parent.smiles] = before
        for candidate in candidates:
            base = _row_base(
                method=method,
                distance_type=distance_type,
                ged_mode=effective_ged_mode,
                parent=parent,
                candidate=candidate,
            )
            after = candidate_pred_cache.get(candidate.smiles)
            if after is None:
                after = predict_with_teacher(teacher, candidate.smiles, parent.label)
                candidate_pred_cache[candidate.smiles] = after
            row = dict(base)
            row.update(
                {
                    "fragment_smiles": "",
                    "match": True,
                    "delete_valid": "",
                    "residual_smiles": candidate.smiles,
                    "p_before": before.get("p_label"),
                    "p_after": after.get("p_label"),
                    "pred_before": before.get("pred_label"),
                    "pred_after": after.get("pred_label"),
                    "cf_drop": (
                        float(before["p_label"]) - float(after["p_label"])
                        if before.get("ok") and after.get("ok")
                        else None
                    ),
                    "error": after.get("error") if not after.get("ok") else None,
                }
            )
            row.update(
                teacher_flip_audit_fields(
                    before.get("pred_label"),
                    after.get("pred_label"),
                    parent.label,
                )
            )
            distance_result = _compute_pair_distance(
                teacher=teacher,
                teacher_path=teacher_path,
                parent_smiles=parent.smiles,
                candidate_smiles=candidate.smiles,
                distance_type=distance_type,
                ged_mode=effective_ged_mode,
                cache=cache,
                delete_candidate=None,
            )
            row.update(
                {
                    "distance": distance_result.get("distance"),
                    "cosine_similarity": distance_result.get("cosine_similarity"),
                    "embedding_ok": distance_result.get("embedding_ok"),
                    "ged_ok": distance_result.get("ged_ok"),
                    "error": row.get("error") or distance_result.get("error"),
                }
            )
            details.append(row)
    cache.flush()
    summaries = build_threshold_summary(
        details,
        method=method,
        distance_type=distance_type,
        ged_mode=effective_ged_mode,
        thresholds=thresholds,
        total_parents=len(parents),
        total_candidates=len(candidates),
        require_flip_only=require_flip_only,
        min_cf_drop=min_cf_drop,
        desired_label=desired_label,
        cache_hit_rate=cache.hit_rate,
    )
    for row in summaries:
        row["num_matched_parents"] = None
        row["match_rate"] = None
        row["num_delete_valid_parents"] = None
        row["delete_valid_rate"] = None
    report = render_report(
        method_name=method,
        distance_type=distance_type,
        ged_mode=effective_ged_mode,
        thresholds=thresholds,
        summaries=summaries,
        total_parents=len(parents),
        total_candidates=len(candidates),
        match_metric=False,
    )
    outputs = _write_outputs(output_dir=output_root, details=details, summaries=summaries, report=report)
    print(
        f"[EVAL_GCF] parents={len(parents)} candidates={len(candidates)} detail_rows={len(details)} "
        f"seconds={time.time() - started:.2f}"
    )
    print(f"[DISTANCE_CACHE] path={cache.path} hits={cache.hits} misses={cache.misses} hit_rate={cache.hit_rate:.4f}")
    if summaries:
        last = summaries[-1]
        print(
            f"[THRESHOLD_SUMMARY] threshold={last['threshold']} close_only={last['close_only_coverage']:.4f} "
            f"close_cf={last['close_cf_coverage']:.4f}"
        )
    return {"details": details, "threshold_summary": summaries, "outputs": outputs}


__all__ = [
    "DETAIL_FIELDS",
    "SUMMARY_FIELDS",
    "build_threshold_summary",
    "canonicalize_smiles",
    "embedding_distance_from_teacher",
    "evaluate_gcf_counterfactual_graphs",
    "evaluate_ours_selected_subgraphs",
    "hard_delete_substructure_any_match",
    "mol_from_smiles",
    "mol_to_labeled_nx_graph",
    "normalized_delete_ged_distance",
    "normalized_networkx_ged_distance",
    "predict_with_teacher",
    "render_report",
]
