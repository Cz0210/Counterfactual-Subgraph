"""Train-only CLEAR/GraphCFE candidate-pool helpers for Mutagenicity.

The official implementation owns graph-predictor and GraphCFE training.  This
module owns the strict project data contract, streaming chemistry projection,
RF-teacher validation, resume state, and artifact audit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
from rdkit import Chem

from src.baselines.clear_mutagenicity_adapter import (
    ATOM_SIDECAR_SCHEMA_VERSION,
    ClearMutagenicityCodecError,
    ClearMutagenicitySchema,
    PreparedMolecule,
    codec_provenance,
    load_strict_cohort,
    project_binary_graph_to_molecule,
)


DATASET = "Mutagenicity"
GENERATOR_METHOD = "CLEAR-GraphCFE"
SOURCE_LABEL = 1
TARGET_LABEL = 0
EXPECTED_MODEL_TRAIN_ROWS = 2885
EXPECTED_MODEL_VAL_ROWS = 355
EXPECTED_GENERATION_PARENT_ROWS = 1448
REQUIRED_OUTPUTS = (
    "raw_generated_candidates.jsonl",
    "invalid_candidates.jsonl",
    "non_target_candidates.jsonl",
    "candidate_pool.jsonl",
    "candidate_universe.jsonl",
    "generation_progress.json",
    "summary.json",
    "run_manifest.json",
    "train_pool_audit.json",
    "_RUN_COMPLETE.json",
)


class ClearMutagenicityTrainPoolError(RuntimeError):
    """Raised when the Phase B strict protocol cannot be satisfied."""


class ClearMutagenicityEmptyPoolError(ClearMutagenicityTrainPoolError):
    """Raised when CLEAR produced no RF-valid target candidates."""


class TeacherProtocol(Protocol):
    available: bool

    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        parent_smiles: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TrainPoolConfig:
    parent_limit: int = 64
    graphpred_epochs: int = 5
    cfe_epochs: int = 5
    generation_chunk_size: int = 16
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 13
    device: str = "cuda"
    resume: bool = True
    expected_model_train_rows: int = EXPECTED_MODEL_TRAIN_ROWS
    expected_model_val_rows: int = EXPECTED_MODEL_VAL_ROWS
    expected_generation_parent_rows: int = EXPECTED_GENERATION_PARENT_ROWS

    def validate_smoke(self) -> None:
        if int(self.parent_limit) != 64:
            raise ValueError(
                "Phase B is intentionally limited to the 64-parent smoke; "
                f"found parent_limit={self.parent_limit}."
            )
        for field_name in (
            "graphpred_epochs",
            "cfe_epochs",
            "generation_chunk_size",
            "batch_size",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if int(self.num_workers) != 0:
            raise ValueError("Phase B generation/training num_workers must be 0.")

    def identity(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeneratedGraph:
    """One lightweight official GraphCFE inference result."""

    parent_id: str
    features: np.ndarray
    adjacency: np.ndarray
    official_pred_before: int | None
    official_pred_after: int | None
    official_prob_before: tuple[float, ...] = ()
    official_prob_after: tuple[float, ...] = ()
    generator_rank: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            yield payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_candidate_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
    return f"CLEAR_MUT_{digest[:20].upper()}"


def cohort_hash(rows: Sequence[PreparedMolecule]) -> str:
    payload = [
        {
            "molecule_id": row.molecule_id,
            "canonical_smiles": row.canonical_smiles,
            "label": int(row.label),
            "split": row.split,
        }
        for row in rows
    ]
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def select_generation_parents(
    rows: Sequence[PreparedMolecule], parent_limit: int
) -> list[PreparedMolecule]:
    ordered = sorted(rows, key=lambda row: row.molecule_id)
    if int(parent_limit) <= 0:
        return ordered
    return ordered[: int(parent_limit)]


def schema_from_mapping(payload: dict[str, Any]) -> ClearMutagenicitySchema:
    fields = {
        "atom_vocabulary": tuple(int(value) for value in payload["atom_vocabulary"]),
        "formal_charge_vocabulary": tuple(
            int(value) for value in payload["formal_charge_vocabulary"]
        ),
        "aromaticity_vocabulary": tuple(
            bool(value) for value in payload["aromaticity_vocabulary"]
        ),
        "bond_type_vocabulary": tuple(
            str(value) for value in payload["bond_type_vocabulary"]
        ),
        "atom_feature_start": int(payload["atom_feature_start"]),
        "atom_feature_end": int(payload["atom_feature_end"]),
        "charge_feature_start": int(payload["charge_feature_start"]),
        "charge_feature_end": int(payload["charge_feature_end"]),
        "aromatic_feature_index": int(payload["aromatic_feature_index"]),
        "node_present_feature_index": int(payload["node_present_feature_index"]),
        "feature_dim": int(payload["feature_dim"]),
        "vocabulary_source": str(
            payload.get("vocabulary_source", "strict_train_only")
        ),
        "max_num_nodes_source": str(
            payload.get(
                "max_num_nodes_source", "strict_train_plus_validation"
            )
        ),
    }
    return ClearMutagenicitySchema(**fields)


def validate_phase_a_data(
    data: Any,
    *,
    expected_train_rows: int,
    expected_val_rows: int,
) -> dict[str, Any]:
    required = (
        "molecule_id_all",
        "canonical_smiles_all",
        "source_split_all",
        "molecule_sidecar_all",
        "feature_schema",
        "atom_sidecar_schema_version",
        "max_num_nodes",
        "labels_all",
    )
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise ClearMutagenicityTrainPoolError(
            f"Phase A CLEAR pickle is missing fields: {missing}"
        )
    if str(data.atom_sidecar_schema_version) != ATOM_SIDECAR_SCHEMA_VERSION:
        raise ClearMutagenicityTrainPoolError(
            "Phase A atom sidecar schema mismatch: "
            f"{data.atom_sidecar_schema_version!r} != "
            f"{ATOM_SIDECAR_SCHEMA_VERSION!r}"
        )
    splits = [str(value) for value in data.source_split_all]
    train_rows = sum(value == "train" for value in splits)
    val_rows = sum(value == "val" for value in splits)
    forbidden = sorted({value for value in splits if value not in {"train", "val"}})
    if forbidden:
        raise ClearMutagenicityTrainPoolError(
            f"Phase A pickle contains forbidden split(s): {forbidden}"
        )
    if train_rows != int(expected_train_rows) or val_rows != int(
        expected_val_rows
    ):
        raise ClearMutagenicityTrainPoolError(
            "Phase A model cohort count mismatch: "
            f"train={train_rows}/{expected_train_rows}, "
            f"val={val_rows}/{expected_val_rows}"
        )
    if int(data.max_num_nodes) != 99:
        raise ClearMutagenicityTrainPoolError(
            f"Expected frozen Phase A max_num_nodes=99, found {data.max_num_nodes}."
        )
    schema = schema_from_mapping(dict(data.feature_schema))
    return {
        "model_train_rows": train_rows,
        "model_val_rows": val_rows,
        "max_num_nodes": int(data.max_num_nodes),
        "feature_dim": int(schema.feature_dim),
        "atom_sidecar_schema_version": str(data.atom_sidecar_schema_version),
        "calibration_loaded": False,
        "test_loaded": False,
    }


def validate_phase_a_splits(
    data: Any,
    split_payload: dict[str, Any],
) -> dict[str, Any]:
    expected_train = [
        index
        for index, split in enumerate(data.source_split_all)
        if str(split) == "train"
    ]
    expected_val = [
        index
        for index, split in enumerate(data.source_split_all)
        if str(split) == "val"
    ]
    for key in ("idx_train_list", "idx_val_list", "idx_test_list"):
        if key not in split_payload or not split_payload[key]:
            raise ClearMutagenicityTrainPoolError(
                f"Phase A split pickle is missing {key} repetitions."
            )
    repetitions = min(
        len(split_payload["idx_train_list"]),
        len(split_payload["idx_val_list"]),
        len(split_payload["idx_test_list"]),
    )
    for repetition in range(repetitions):
        train_indices = [
            int(value)
            for value in np.asarray(
                split_payload["idx_train_list"][repetition]
            ).reshape(-1)
        ]
        val_indices = [
            int(value)
            for value in np.asarray(
                split_payload["idx_val_list"][repetition]
            ).reshape(-1)
        ]
        test_alias_indices = [
            int(value)
            for value in np.asarray(
                split_payload["idx_test_list"][repetition]
            ).reshape(-1)
        ]
        if train_indices != expected_train:
            raise ClearMutagenicityTrainPoolError(
                f"Official train split repetition {repetition} is not the "
                "strict project train cohort."
            )
        if val_indices != expected_val:
            raise ClearMutagenicityTrainPoolError(
                f"Official val split repetition {repetition} is not the "
                "strict project validation cohort."
            )
        if test_alias_indices != expected_val:
            raise ClearMutagenicityTrainPoolError(
                f"Official test-loader repetition {repetition} must alias "
                "project validation; external test data is forbidden."
            )
    if split_payload.get("calibration_loaded") is not False:
        raise ClearMutagenicityTrainPoolError(
            "Phase A split pickle must record calibration_loaded=false."
        )
    if split_payload.get("test_loaded") is not False:
        raise ClearMutagenicityTrainPoolError(
            "Phase A split pickle must record test_loaded=false."
        )
    return {
        "official_split_repetitions_checked": repetitions,
        "official_train_indices_exact": True,
        "official_val_indices_exact": True,
        "official_test_loader_is_validation_alias": True,
    }


def validate_generation_mapping(
    data: Any, parents: Sequence[PreparedMolecule]
) -> dict[str, int]:
    index_by_id = {
        str(molecule_id): index
        for index, molecule_id in enumerate(data.molecule_id_all)
    }
    if len(index_by_id) != len(data.molecule_id_all):
        raise ClearMutagenicityTrainPoolError(
            "Phase A pickle has duplicate molecule_id entries."
        )
    mapping: dict[str, int] = {}
    for parent in parents:
        if parent.molecule_id not in index_by_id:
            raise ClearMutagenicityTrainPoolError(
                f"Generation parent is absent from Phase A pickle: "
                f"{parent.molecule_id}"
            )
        index = index_by_id[parent.molecule_id]
        if str(data.source_split_all[index]) != "train":
            raise ClearMutagenicityTrainPoolError(
                f"Generation parent is not a train row: {parent.molecule_id}"
            )
        if str(data.canonical_smiles_all[index]) != parent.canonical_smiles:
            raise ClearMutagenicityTrainPoolError(
                f"Generation parent SMILES mismatch: {parent.molecule_id}"
            )
        mapping[parent.molecule_id] = int(index)
    return mapping


def teacher_probabilities(
    teacher: TeacherProtocol, smiles: str
) -> tuple[int, float, float]:
    target = teacher.score_smiles(smiles, label=TARGET_LABEL)
    source = teacher.score_smiles(smiles, label=SOURCE_LABEL)
    if not bool(target.get("teacher_result_ok")) or not bool(
        source.get("teacher_result_ok")
    ):
        raise ClearMutagenicityTrainPoolError(
            "RF teacher failed for molecule: "
            f"{smiles!r}; target_reason={target.get('teacher_reason')}, "
            f"source_reason={source.get('teacher_reason')}"
        )
    pred0 = int(target["teacher_label"])
    pred1 = int(source["teacher_label"])
    if pred0 != pred1:
        raise ClearMutagenicityTrainPoolError(
            f"RF teacher returned inconsistent labels for {smiles!r}."
        )
    p0 = float(target["teacher_prob"])
    p1 = float(source["teacher_prob"])
    if not all(math.isfinite(value) for value in (p0, p1)):
        raise ClearMutagenicityTrainPoolError(
            f"RF teacher returned non-finite probabilities for {smiles!r}."
        )
    return pred0, p0, p1


def _empty_parts(output_dir: Path) -> dict[str, Path]:
    return {
        "raw": output_dir / "raw_generated_candidates.jsonl.part",
        "invalid": output_dir / "invalid_candidates.jsonl.part",
        "non_target": output_dir / "non_target_candidates.jsonl.part",
        "pool": output_dir / "candidate_pool.jsonl.part",
    }


def _final_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "raw": output_dir / "raw_generated_candidates.jsonl",
        "invalid": output_dir / "invalid_candidates.jsonl",
        "non_target": output_dir / "non_target_candidates.jsonl",
        "pool": output_dir / "candidate_pool.jsonl",
    }


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def _truncate_jsonl(path: Path, rows: int) -> None:
    existing = read_jsonl(path) if path.exists() else []
    if len(existing) < int(rows):
        raise ClearMutagenicityTrainPoolError(
            f"Resume file {path} has {len(existing)} rows, expected at least {rows}."
        )
    write_jsonl(path, existing[: int(rows)])


def _record_from_generated(
    *,
    generated: GeneratedGraph,
    parent: PreparedMolecule,
    parent_sidecar: dict[str, Any],
    schema: ClearMutagenicitySchema,
    teacher: TeacherProtocol,
    seed: int,
    chunk_index: int,
) -> tuple[dict[str, Any], str]:
    source_pred, source_p0, source_p1 = teacher_probabilities(
        teacher, parent.canonical_smiles
    )
    if source_pred != SOURCE_LABEL:
        raise ClearMutagenicityTrainPoolError(
            f"Generation source is not RF teacher label 1: {parent.molecule_id}"
        )
    codec = project_binary_graph_to_molecule(
        features=generated.features,
        adjacency=generated.adjacency,
        schema=schema,
        parent_sidecar=parent_sidecar,
        atom_attribute_mode="generated",
        require_connected=True,
    )
    official_flip = (
        generated.official_pred_before == SOURCE_LABEL
        and generated.official_pred_after == TARGET_LABEL
    )
    row: dict[str, Any] = {
        "candidate_id": None,
        "canonical_smiles": codec.canonical_smiles,
        "raw_smiles": codec.canonical_smiles,
        "source_parent_id": parent.molecule_id,
        "source_parent_smiles": parent.canonical_smiles,
        "source_split": "train",
        "generator_method": GENERATOR_METHOD,
        "generator_rank": int(generated.generator_rank),
        "generator_score": None,
        "native_run_id": "clear_mutagenicity_phase_b_smoke",
        "rdkit_parse_ok": bool(codec.ok),
        "codec_ok": bool(codec.ok),
        "codec_error_type": codec.error_type,
        "codec_error": codec.error,
        "official_pred_before": generated.official_pred_before,
        "official_pred_after": generated.official_pred_after,
        "official_prob_before": list(generated.official_prob_before),
        "official_prob_after": list(generated.official_prob_after),
        "official_clear_flip": bool(official_flip),
        "source_teacher_pred": int(source_pred),
        "source_teacher_prob_0": float(source_p0),
        "source_teacher_prob_1": float(source_p1),
        "teacher_pred": None,
        "teacher_prob_0": None,
        "teacher_prob_1": None,
        "teacher_target_ok": False,
        "strict_flip": False,
        "num_atoms": 0,
        "num_bonds": 0,
        "seed": int(seed),
        "chunk_index": int(chunk_index),
        "projection_provenance": {
            **codec_provenance(),
            "atom_attribute_mode": "source_anchored_generated",
            "unchanged_parent_atoms_inherit_explicit_h_chirality": True,
            "changed_atom_policy": "ambiguous_generated_atom_hydrogen_state",
        },
        **codec.to_dict(),
    }
    if not codec.ok or codec.molecule is None or not codec.canonical_smiles:
        return row, "invalid"
    canonical = str(codec.canonical_smiles)
    candidate_pred, candidate_p0, candidate_p1 = teacher_probabilities(
        teacher, canonical
    )
    row.update(
        {
            "candidate_id": stable_candidate_id(canonical),
            "canonical_smiles": canonical,
            "raw_smiles": canonical,
            "teacher_pred": int(candidate_pred),
            "teacher_prob_0": float(candidate_p0),
            "teacher_prob_1": float(candidate_p1),
            "teacher_target_ok": candidate_pred == TARGET_LABEL,
            "strict_flip": (
                source_pred == SOURCE_LABEL and candidate_pred == TARGET_LABEL
            ),
            "num_atoms": int(codec.molecule.GetNumAtoms()),
            "num_bonds": int(codec.molecule.GetNumBonds()),
        }
    )
    return row, "pool" if row["strict_flip"] else "non_target"


def candidate_universe_from_pool(
    pool_rows: Sequence[dict[str, Any]],
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
        representative = min(
            occurrences,
            key=lambda row: (
                int(row.get("generator_rank") or 0),
                str(row["source_parent_id"]),
            ),
        )
        universe.append(
            {
                "candidate_id": stable_candidate_id(canonical),
                "canonical_smiles": canonical,
                "source_parent_ids": source_ids,
                "source_parent_count": len(source_ids),
                "occurrence_count": len(occurrences),
                "generator_method": GENERATOR_METHOD,
                "generator_rank": int(
                    representative.get("generator_rank") or 1
                ),
                "projection_provenance": representative.get(
                    "projection_provenance"
                ),
                "teacher_pred": TARGET_LABEL,
                "teacher_prob_0": representative.get("teacher_prob_0"),
                "teacher_prob_1": representative.get("teacher_prob_1"),
                "teacher_target_ok": True,
                "rdkit_parse_ok": True,
                "num_atoms": representative.get("num_atoms"),
                "num_bonds": representative.get("num_bonds"),
            }
        )
    return universe


def run_streaming_generation(
    *,
    output_dir: str | Path,
    parents: Sequence[PreparedMolecule],
    data: Any,
    schema: ClearMutagenicitySchema,
    teacher: TeacherProtocol,
    generate_chunk: Callable[
        [Sequence[PreparedMolecule], Sequence[int], int],
        Sequence[GeneratedGraph],
    ],
    config: TrainPoolConfig,
    config_fingerprint: str,
    model_checkpoint_hash: str,
) -> dict[str, Any]:
    """Generate, decode, score, and persist one deterministic chunk at a time."""

    config.validate_smoke()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parts = _empty_parts(destination)
    finals = _final_paths(destination)
    progress_path = destination / "generation_progress.json"
    mapping = validate_generation_mapping(data, parents)
    parent_ids = [row.molecule_id for row in parents]
    selected_hash = cohort_hash(parents)
    expected_resume = {
        "config_fingerprint": config_fingerprint,
        "selected_parent_hash": selected_hash,
        "model_checkpoint_hash": model_checkpoint_hash,
        "generation_chunk_size": int(config.generation_chunk_size),
        "selected_parent_ids": parent_ids,
    }
    row_counts = {name: 0 for name in parts}
    next_offset = 0
    completed_chunks = 0

    def completed_summary() -> dict[str, Any]:
        raw_rows = read_jsonl(finals["raw"])
        pool_rows = read_jsonl(finals["pool"])
        universe_rows = read_jsonl(destination / "candidate_universe.jsonl")
        if len(raw_rows) != len(parents):
            raise ClearMutagenicityTrainPoolError(
                "Completed generation raw row count differs from selected parents."
            )
        if not pool_rows or not universe_rows:
            raise ClearMutagenicityEmptyPoolError(
                "Completed generation has an empty candidate pool/universe."
            )
        source_ids = {
            str(row["source_parent_id"]) for row in pool_rows
        }
        return {
            "selected_generation_parents": len(parents),
            "generation_chunk_size": int(config.generation_chunk_size),
            "completed_chunk_count": int(
                math.ceil(len(parents) / int(config.generation_chunk_size))
            ),
            "raw_generated_rows": len(raw_rows),
            "invalid_candidate_rows": _count_jsonl(finals["invalid"]),
            "non_target_candidate_rows": _count_jsonl(
                finals["non_target"]
            ),
            "candidate_pool_rows": len(pool_rows),
            "canonical_unique_candidates": len(universe_rows),
            "candidate_source_parent_rows": len(source_ids),
            "source_parent_coverage": len(source_ids) / len(parents),
            "chunk_resume_duplicate_rows": 0,
            "generation_run_complete": True,
        }

    if progress_path.exists():
        if not config.resume:
            raise FileExistsError(
                "Generation progress exists and --no-resume was requested."
            )
        progress = read_json(progress_path)
        for key, expected in expected_resume.items():
            if progress.get(key) != expected:
                raise ClearMutagenicityTrainPoolError(
                    f"Generation resume mismatch for {key}: "
                    f"{progress.get(key)!r} != {expected!r}"
                )
        next_offset = int(progress["next_parent_offset"])
        completed_chunks = int(progress["completed_chunk_count"])
        row_counts = {
            name: int(progress["part_row_counts"].get(name, 0))
            for name in parts
        }
        if progress.get("run_complete") is True:
            required_completed = (
                *finals.values(),
                destination / "candidate_universe.jsonl",
            )
            missing_completed = [
                str(path)
                for path in required_completed
                if not path.is_file()
            ]
            if missing_completed:
                raise ClearMutagenicityTrainPoolError(
                    "Completed generation progress is missing final artifacts: "
                    f"{missing_completed}"
                )
            return completed_summary()
        for name, path in parts.items():
            if not path.exists() and finals[name].exists():
                os.replace(finals[name], path)
            _truncate_jsonl(path, row_counts[name])
    else:
        for path in parts.values():
            if path.exists():
                raise ClearMutagenicityTrainPoolError(
                    f"Stale generation part file without progress: {path}"
                )
            write_jsonl(path, [])

    chunk_size = int(config.generation_chunk_size)
    for offset in range(next_offset, len(parents), chunk_size):
        chunk = list(parents[offset : offset + chunk_size])
        indices = [mapping[row.molecule_id] for row in chunk]
        generated_rows = list(
            generate_chunk(chunk, indices, completed_chunks)
        )
        if [row.parent_id for row in generated_rows] != [
            row.molecule_id for row in chunk
        ]:
            raise ClearMutagenicityTrainPoolError(
                "Official generation changed parent order or row count."
            )
        buckets: dict[str, list[dict[str, Any]]] = {
            name: [] for name in parts
        }
        for generated, parent, data_index in zip(
            generated_rows, chunk, indices, strict=True
        ):
            record, classification = _record_from_generated(
                generated=generated,
                parent=parent,
                parent_sidecar=dict(data.molecule_sidecar_all[data_index]),
                schema=schema,
                teacher=teacher,
                seed=int(config.seed),
                chunk_index=completed_chunks,
            )
            buckets["raw"].append(record)
            buckets[classification].append(record)
        for name, rows in buckets.items():
            row_counts[name] += append_jsonl(parts[name], rows)
        completed_chunks += 1
        next_offset = offset + len(chunk)
        write_json(
            progress_path,
            {
                **expected_resume,
                "completed_chunk_count": completed_chunks,
                "next_parent_offset": next_offset,
                "selected_parent_count": len(parents),
                "part_row_counts": row_counts,
                "updated_at": utc_now(),
                "run_complete": False,
            },
        )

    for name, part in parts.items():
        os.replace(part, finals[name])
    raw_rows = read_jsonl(finals["raw"])
    pool_rows = read_jsonl(finals["pool"])
    universe = candidate_universe_from_pool(pool_rows)
    write_jsonl(destination / "candidate_universe.jsonl", universe)
    if not pool_rows or not universe:
        write_json(
            progress_path,
            {
                **expected_resume,
                "completed_chunk_count": completed_chunks,
                "next_parent_offset": len(parents),
                "selected_parent_count": len(parents),
                "part_row_counts": row_counts,
                "updated_at": utc_now(),
                "run_complete": False,
                "error": "empty_candidate_pool",
            },
        )
        raise ClearMutagenicityEmptyPoolError(
            "CLEAR generation completed but the strict RF-valid candidate pool "
            "or candidate universe is empty."
        )
    write_json(
        progress_path,
        {
            **expected_resume,
            "completed_chunk_count": completed_chunks,
            "next_parent_offset": len(parents),
            "selected_parent_count": len(parents),
            "part_row_counts": row_counts,
            "raw_generated_rows": len(raw_rows),
            "candidate_pool_rows": len(pool_rows),
            "candidate_universe_rows": len(universe),
            "updated_at": utc_now(),
            "run_complete": True,
        },
    )
    return completed_summary()


def _canonical_valid(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == smiles
    )


def audit_train_pool(
    *,
    run_dir: str | Path,
    generation_csv: str | Path,
    expected_model_train_rows: int = EXPECTED_MODEL_TRAIN_ROWS,
    expected_model_val_rows: int = EXPECTED_MODEL_VAL_ROWS,
    expected_generation_parent_rows: int = EXPECTED_GENERATION_PARENT_ROWS,
    expected_selected_parents: int = 64,
    require_complete: bool = True,
    teacher: TeacherProtocol | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    progress = read_json(root / "generation_progress.json")
    pool = read_jsonl(root / "candidate_pool.jsonl")
    universe = read_jsonl(root / "candidate_universe.jsonl")
    raw = read_jsonl(root / "raw_generated_candidates.jsonl")
    parents = load_strict_cohort(
        generation_csv,
        expected_split="train",
        expected_label=SOURCE_LABEL,
        expected_rows=int(expected_generation_parent_rows),
    )
    selected = select_generation_parents(parents, expected_selected_parents)
    selected_ids = {row.molecule_id for row in selected}
    pool_parent_ids = {str(row["source_parent_id"]) for row in pool}
    if not pool_parent_ids.issubset(selected_ids):
        extra = sorted(pool_parent_ids - selected_ids)
        raise AssertionError(
            f"Candidate pool includes parents outside selected train cohort: {extra[:5]}"
        )
    if len(raw) != int(expected_selected_parents):
        raise AssertionError(
            f"Raw generation rows mismatch: {len(raw)} != {expected_selected_parents}"
        )
    if len({str(row["source_parent_id"]) for row in raw}) != len(raw):
        raise AssertionError("Chunk/resume produced duplicate generation parents.")
    if not pool or not universe:
        raise AssertionError("Candidate pool and universe must be non-empty.")
    canonical_values = [str(row["canonical_smiles"]) for row in universe]
    if len(canonical_values) != len(set(canonical_values)):
        raise AssertionError("Candidate universe is not canonical-SMILES unique.")
    if any(not _canonical_valid(smiles) for smiles in canonical_values):
        raise AssertionError("Candidate universe contains invalid/noncanonical SMILES.")
    for row in universe:
        if int(row.get("teacher_pred")) != TARGET_LABEL:
            raise AssertionError("Candidate universe contains teacher_pred != 0.")
        if not bool(row.get("teacher_target_ok")):
            raise AssertionError("Candidate universe contains non-target candidate.")
        canonical = str(row["canonical_smiles"])
        if str(row["candidate_id"]) != stable_candidate_id(canonical):
            raise AssertionError("Candidate ID hash mismatch.")
        occurrences = [
            candidate
            for candidate in pool
            if str(candidate["canonical_smiles"]) == canonical
        ]
        source_ids = sorted(
            {str(candidate["source_parent_id"]) for candidate in occurrences}
        )
        if int(row["source_parent_count"]) != len(source_ids):
            raise AssertionError("source_parent_count mismatch.")
        if int(row["occurrence_count"]) != len(occurrences):
            raise AssertionError("occurrence_count mismatch.")
        if list(row["source_parent_ids"]) != source_ids:
            raise AssertionError("source_parent_ids mismatch.")
        if teacher is not None:
            prediction, probability_0, probability_1 = teacher_probabilities(
                teacher, canonical
            )
            if prediction != TARGET_LABEL:
                raise AssertionError(
                    "RF teacher rescore rejected candidate universe molecule."
                )
            for key, expected in (
                ("teacher_prob_0", probability_0),
                ("teacher_prob_1", probability_1),
            ):
                stored = row.get(key)
                if stored is not None and not math.isclose(
                    float(stored),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise AssertionError(f"{key} differs from RF teacher rescore.")
    if int(summary.get("model_train_rows", -1)) != int(
        expected_model_train_rows
    ):
        raise AssertionError("Graph predictor/CFE model train count mismatch.")
    if int(summary.get("model_val_rows", -1)) != int(expected_model_val_rows):
        raise AssertionError("Graph predictor/CFE validation count mismatch.")
    if int(summary.get("selected_generation_parents", -1)) != int(
        expected_selected_parents
    ):
        raise AssertionError("Selected generation parent count mismatch.")
    for name, value in dict(manifest.get("inputs", {})).items():
        basename = Path(str(value)).name.lower()
        if any(
            token in basename
            for token in (
                "calibration_source",
                "calibration_target",
                "test_source",
                "test_target",
            )
        ):
            raise AssertionError(
                f"Calibration/test input leaked into Phase B: {name}={value}"
            )
    for payload in (summary, manifest):
        if payload.get("calibration_loaded") is not False:
            raise AssertionError("calibration_loaded must be false.")
        if payload.get("test_loaded") is not False:
            raise AssertionError("test_loaded must be false.")
    if progress.get("run_complete") is not True:
        raise AssertionError("Generation progress is incomplete.")
    if require_complete:
        if summary.get("run_complete") is not True:
            raise AssertionError("Summary is incomplete.")
        if manifest.get("run_complete") is not True:
            raise AssertionError("Manifest is incomplete.")
        if not (root / "_RUN_COMPLETE.json").is_file():
            raise AssertionError("Missing _RUN_COMPLETE.json.")
    recomputed_coverage = len(pool_parent_ids) / len(selected)
    if not math.isclose(
        float(summary["source_parent_coverage"]),
        recomputed_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("source_parent_coverage mismatch.")
    result = {
        "model_train_rows": int(summary["model_train_rows"]),
        "model_val_rows": int(summary["model_val_rows"]),
        "generation_source_rows": len(parents),
        "selected_generation_parents": len(selected),
        "candidate_pool_rows": len(pool),
        "candidate_universe_rows": len(universe),
        "candidate_source_parent_rows": len(pool_parent_ids),
        "source_parent_coverage_recomputed": recomputed_coverage,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "chunk_resume_duplicate_rows": 0,
        "teacher_rescored_candidate_rows": len(universe)
        if teacher is not None
        else 0,
        "run_complete": True,
        "audit_passed": True,
    }
    write_json(root / "train_pool_audit.json", result)
    return result


__all__ = [
    "ClearMutagenicityEmptyPoolError",
    "ClearMutagenicityTrainPoolError",
    "GeneratedGraph",
    "TrainPoolConfig",
    "audit_train_pool",
    "candidate_universe_from_pool",
    "cohort_hash",
    "run_streaming_generation",
    "schema_from_mapping",
    "select_generation_parents",
    "sha256_file",
    "stable_candidate_id",
    "teacher_probabilities",
    "validate_generation_mapping",
    "validate_phase_a_data",
    "validate_phase_a_splits",
    "write_json",
]
