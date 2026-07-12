#!/usr/bin/env python3
"""Quick HIV/SMILES recourse and CAMC comparison for counterfactual baselines."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chem.deletion import delete_fragment_from_parent  # noqa: E402
from src.chem.substructure import is_parent_substructure  # noqa: E402
from src.eval.flip_semantics import teacher_strict_flip  # noqa: E402
from src.rewards.reward_calculator import (  # noqa: E402
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)

try:  # pragma: no cover - depends on runtime env
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdFMCS
except ImportError:  # pragma: no cover - depends on runtime env
    Chem = None
    DataStructs = None
    RDLogger = None
    rdFingerprintGenerator = None
    rdFMCS = None

try:  # pragma: no cover - optional runtime dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None


SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "canonical_smiles", "mol", "molecule")
LABEL_COLUMN_CANDIDATES = ("label", "y", "HIV_active", "active")
FULLGRAPH_SMILES_COLUMNS = ("smiles", "counterfactual_smiles", "selected_smiles", "fullgraph_smiles")
FRAGMENT_KEYS = (
    "final_fragment",
    "core_fragment",
    "fragment_smiles",
    "fragment",
    "selected_fragment",
    "raw_fragment",
)
COMPARISON_FIELDS = [
    "method",
    "target_label",
    "k",
    "theta",
    "num_inputs",
    "coverage",
    "coverage_theta",
    "coverage_unconstrained_flip",
    "median_cost",
    "mean_cost",
    "median_cost_covered_only",
    "mean_cf_drop",
    "mean_cf_drop_covered_only",
    "flip_rate",
    "valid_recourse_rate",
    "runtime_sec",
]
PER_INPUT_FIELDS = [
    "method",
    "target_label",
    "k",
    "theta",
    "input_idx",
    "input_row_index",
    "input_id",
    "parent_smiles",
    "parent_label",
    "p_before",
    "before_teacher_label",
    "recourse_smiles",
    "fragment_rank",
    "fragment_smiles",
    "candidate_rank",
    "candidate_index",
    "candidate_label",
    "p_after",
    "after_teacher_label",
    "cf_drop",
    "cf_flip",
    "cost",
    "proxy_edit",
    "valid_recourse",
    "covered",
    "coverage_theta",
    "coverage_unconstrained_flip",
    "reason",
]
OURS_ACTION_FIELDS = [
    "method",
    "target_label",
    "input_idx",
    "input_row_index",
    "input_id",
    "parent_smiles",
    "parent_label",
    "fragment_rank",
    "fragment",
    "match_ok",
    "delete_ok",
    "valid_after",
    "recourse_smiles",
    "distance_proxy",
    "proxy_edit",
    "p_before",
    "p_after",
    "before_teacher_label",
    "after_teacher_label",
    "cf_drop",
    "cf_flip",
    "failure_reason",
]
CAMC_OURS_MOTIF_FIELDS = [
    "method",
    "action_source",
    "motif_rank",
    "motif_smiles",
    "source_fragment_rank",
    "source",
    "atom_count",
    "bond_count",
    "valid_motif",
    "canonical_motif_smiles",
    "failure_reason",
]
CAMC_FULLGRAPH_POOL_FIELDS = [
    "method",
    "input_idx",
    "parent_smiles",
    "nearest_fullgraph_rank",
    "nearest_fullgraph_smiles",
    "nearest_distance_proxy",
    "fullgraph_cf_flip",
    "mcs_num_atoms",
    "mcs_num_bonds",
    "deleted_motif_smiles",
    "canonical_motif_smiles",
    "motif_atom_count",
    "motif_bond_count",
    "component_count",
    "valid_motif",
    "failure_reason",
]
CAMC_SELECTED_MOTIF_FIELDS = [
    "method",
    "action_source",
    "selection_rank",
    "motif_rank",
    "motif_smiles",
    "canonical_motif_smiles",
    "atom_count",
    "bond_count",
    "valid_motif",
    "source",
    "source_fragment_rank",
    "source_input_count",
    "source_input_indices",
    "selection_mode",
    "selection_score",
    "selection_cf_value",
    "selection_coverage_gain",
    "selection_redundancy",
    "selection_size_penalty",
    "failure_reason",
]


@dataclass(slots=True)
class TeacherScore:
    ok: bool
    p_target: float | None
    teacher_label: int | None
    reason: str


@dataclass(slots=True)
class BasicMoleculeRecord:
    row_index: int
    molecule_id: str
    smiles: str
    canonical_smiles: str
    label: int
    atom_count: int
    bond_count: int


@dataclass(slots=True)
class MoleculeRecord:
    row_index: int
    molecule_id: str
    smiles: str
    canonical_smiles: str
    label: int
    atom_count: int
    bond_count: int
    p_target: float | None
    teacher_label: int | None
    teacher_ok: bool
    teacher_reason: str


@dataclass(slots=True)
class FragmentRecord:
    rank: int
    fragment: str
    source: str


@dataclass(slots=True)
class FullgraphRecord:
    method: str
    rank: int
    source_index: int
    source_id: str
    source_path: str
    smiles: str
    canonical_smiles: str
    label: int | None
    atom_count: int
    bond_count: int
    p_target: float | None
    teacher_label: int | None
    teacher_ok: bool
    teacher_reason: str
    score: float | None = None


@dataclass(slots=True)
class MCSResult:
    ok: bool
    cost: float | None
    proxy_edit: float | None
    mcs_atoms: int | None
    mcs_bonds: int | None
    smarts: str | None
    canceled: bool
    reason: str
    elapsed_sec: float


class FlushStreamHandler(logging.StreamHandler):
    """Logging handler that flushes after every record for Slurm tail -f."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    logger = logging.getLogger("hiv_quick_compare")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = FlushStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    external_tee = os.environ.get("HIV_COMPARE_EXTERNAL_TEE", "").strip() == "1"
    if not external_tee:
        file_handler = logging.FileHandler(out_dir / "progress.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        (out_dir / "progress.log").touch(exist_ok=True)
    return logger


def configure_rdkit_warnings(*, suppress: bool, logger: logging.Logger) -> None:
    if not suppress:
        logger.info("RDKit warning suppression disabled by --no-suppress-rdkit-warnings")
        return
    if RDLogger is None:
        logger.info("RDKit RDLogger unavailable; cannot suppress rdApp.warning")
        return
    RDLogger.DisableLog("rdApp.warning")
    logger.info("RDKit rdApp.warning logs suppressed")


@contextmanager
def timed_stage(name: str, logger: logging.Logger, runtime_by_stage: dict[str, float]) -> Iterator[None]:
    logger.info("[stage:start] %s", name)
    start = time.time()
    try:
        yield
    except Exception:
        elapsed = time.time() - start
        runtime_by_stage[name] = runtime_by_stage.get(name, 0.0) + elapsed
        logger.exception("[stage:failed] %s elapsed_sec=%.3f", name, elapsed)
        raise
    else:
        elapsed = time.time() - start
        runtime_by_stage[name] = runtime_by_stage.get(name, 0.0) + elapsed
        logger.info("[stage:end] %s elapsed_sec=%.3f", name, elapsed)


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None,
    disable_tqdm: bool,
) -> Iterable[Any]:
    if disable_tqdm or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, file=sys.stdout)


class RFTeacher:
    """Small sklearn-style teacher wrapper with project-loader priority."""

    def __init__(self, teacher_path: str | Path) -> None:
        self.teacher_path = Path(teacher_path).expanduser().resolve()
        bundle = self._load_bundle(self.teacher_path)
        self.model = bundle["model"]
        self.radius = int(bundle["fingerprint_radius"])
        self.n_bits = int(bundle["fingerprint_bits"])
        raw_classes = bundle.get("class_labels")
        if raw_classes is None:
            raw_classes = getattr(self.model, "classes_", (0, 1))
        self.classes = tuple(int(label) for label in raw_classes)
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}

    @staticmethod
    def _load_bundle(path: Path) -> dict[str, Any]:
        try:
            return load_oracle_bundle(path)
        except Exception as project_exc:
            try:
                import joblib  # type: ignore

                payload = joblib.load(path)
            except Exception:
                try:
                    with path.open("rb") as handle:
                        payload = pickle.load(handle)
                except Exception as pickle_exc:
                    raise RuntimeError(
                        "Could not load teacher bundle via project loader, joblib, or pickle: "
                        f"project_error={project_exc}; pickle_error={pickle_exc}"
                    ) from pickle_exc

        if isinstance(payload, dict):
            bundle = payload
        else:
            bundle = {
                "model": payload,
                "fingerprint_radius": getattr(payload, "fingerprint_radius", None),
                "fingerprint_bits": getattr(payload, "fingerprint_bits", None),
                "class_labels": getattr(payload, "classes_", None),
            }

        missing = [
            key
            for key in ("model", "fingerprint_radius", "fingerprint_bits")
            if bundle.get(key) is None
        ]
        if missing:
            raise ValueError(f"Teacher bundle is missing required keys: {missing}")
        if not hasattr(bundle["model"], "predict_proba"):
            raise ValueError("Teacher model must expose predict_proba(...).")
        return bundle

    def score(self, smiles: str, *, target_label: int) -> TeacherScore:
        prepared = prepare_smiles_for_oracle(smiles)
        if prepared is None:
            return TeacherScore(False, None, None, "teacher_input_prepare_failed")
        if prepared == "":
            return TeacherScore(False, None, None, "teacher_input_empty_after_prepare")

        fingerprint = smiles_to_morgan_array(
            prepared,
            radius=self.radius,
            n_bits=self.n_bits,
            clean_dummy_atoms=False,
        )
        if fingerprint is None:
            return TeacherScore(False, None, None, "teacher_fingerprint_failed")

        try:
            probabilities = np.asarray(
                self.model.predict_proba(fingerprint.reshape(1, -1))[0],
                dtype=np.float64,
            )
        except Exception as exc:  # pragma: no cover - model-specific
            return TeacherScore(False, None, None, f"teacher_predict_failed:{exc}")

        target_index = self.class_to_index.get(int(target_label))
        if target_index is None or target_index >= probabilities.size:
            return TeacherScore(False, None, None, f"target_label_missing:{target_label}")
        if probabilities.ndim != 1 or probabilities.size == 0:
            return TeacherScore(False, None, None, "teacher_probabilities_invalid")

        pred_index = int(np.argmax(probabilities))
        if pred_index < len(self.classes):
            teacher_label = int(self.classes[pred_index])
        else:
            teacher_label = pred_index
        return TeacherScore(
            ok=True,
            p_target=float(probabilities[target_index]),
            teacher_label=teacher_label,
            reason="ok",
        )


class DistanceProxy:
    """RDKit MCS proxy distance with cache and diagnostics."""

    def __init__(self, *, timeout_sec: int = 1) -> None:
        if Chem is None or rdFMCS is None:
            raise RuntimeError("RDKit is required for MCS proxy distance.")
        self.timeout_sec = int(timeout_sec)
        self._mol_cache: dict[str, Any] = {}
        self._mcs_cache: dict[tuple[str, str], MCSResult] = {}
        self.mcs_calls = 0
        self.mcs_timeout_count = 0
        self.mcs_failed_count = 0
        self.mcs_time_sum = 0.0
        self.mcs_time_max = 0.0

    def mol(self, smiles: str) -> Any | None:
        if smiles not in self._mol_cache:
            self._mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        return self._mol_cache[smiles]

    def find_mcs(self, left_smiles: str, right_smiles: str) -> MCSResult:
        key = tuple(sorted((left_smiles, right_smiles)))
        cached = self._mcs_cache.get(key)
        if cached is not None:
            return cached

        left_mol = self.mol(left_smiles)
        right_mol = self.mol(right_smiles)
        if left_mol is None or right_mol is None:
            result = MCSResult(False, None, None, None, None, None, False, "distance_parse_failed", 0.0)
            self._mcs_cache[key] = result
            self.mcs_failed_count += 1
            return result

        start = time.time()
        self.mcs_calls += 1
        try:
            mcs = rdFMCS.FindMCS(
                [left_mol, right_mol],
                timeout=self.timeout_sec,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
                matchValences=False,
            )
            elapsed = time.time() - start
            self.mcs_time_sum += elapsed
            self.mcs_time_max = max(self.mcs_time_max, elapsed)
            canceled = bool(getattr(mcs, "canceled", False))
            if canceled:
                self.mcs_timeout_count += 1
            mcs_atoms = int(mcs.numAtoms)
            mcs_bonds = int(mcs.numBonds)
            smarts = str(mcs.smartsString or "")
        except Exception as exc:  # pragma: no cover - RDKit-specific
            elapsed = time.time() - start
            self.mcs_time_sum += elapsed
            self.mcs_time_max = max(self.mcs_time_max, elapsed)
            self.mcs_failed_count += 1
            result = MCSResult(False, None, None, None, None, None, False, f"mcs_failed:{exc}", elapsed)
            self._mcs_cache[key] = result
            return result

        atoms1 = int(left_mol.GetNumAtoms())
        atoms2 = int(right_mol.GetNumAtoms())
        bonds1 = int(left_mol.GetNumBonds())
        bonds2 = int(right_mol.GetNumBonds())
        proxy_edit = (atoms1 - mcs_atoms) + (atoms2 - mcs_atoms) + (bonds1 - mcs_bonds) + (bonds2 - mcs_bonds)
        denominator = max(1, atoms1 + atoms2 + bonds1 + bonds2)
        result = MCSResult(
            ok=True,
            cost=float(proxy_edit) / float(denominator),
            proxy_edit=float(proxy_edit),
            mcs_atoms=mcs_atoms,
            mcs_bonds=mcs_bonds,
            smarts=smarts,
            canceled=canceled,
            reason="mcs_timeout" if canceled else "ok",
            elapsed_sec=elapsed,
        )
        self._mcs_cache[key] = result
        return result

    def distance(self, left_smiles: str, right_smiles: str) -> dict[str, Any]:
        if left_smiles == right_smiles:
            return {
                "ok": True,
                "cost": 0.0,
                "proxy_edit": 0.0,
                "mcs_atoms": None,
                "mcs_bonds": None,
                "reason": "identical",
            }
        mcs = self.find_mcs(left_smiles, right_smiles)
        return {
            "ok": mcs.ok,
            "cost": mcs.cost,
            "proxy_edit": mcs.proxy_edit,
            "mcs_atoms": mcs.mcs_atoms,
            "mcs_bonds": mcs.mcs_bonds,
            "reason": mcs.reason,
        }

    def statistics(self) -> dict[str, Any]:
        avg = self.mcs_time_sum / float(self.mcs_calls) if self.mcs_calls else 0.0
        return {
            "mcs_calls": int(self.mcs_calls),
            "mcs_timeout_count": int(self.mcs_timeout_count),
            "mcs_failed_count": int(self.mcs_failed_count),
            "avg_mcs_time": float(avg),
            "max_mcs_time": float(self.mcs_time_max),
            "mcs_cache_size": int(len(self._mcs_cache)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Ignored config path for Slurm wrapper parity.")
    parser.add_argument("--set", action="append", default=[], help="Ignored dotted override for Slurm parity.")
    parser.add_argument("--hiv-csv", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--target-label", required=True, type=int)
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--ours-selected-dir", required=True, type=Path)
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--theta-list", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--max-inputs", type=int, default=None)
    parser.add_argument("--max-gt-candidates", type=int, default=2000)
    parser.add_argument("--mcs-timeout-sec", type=int, default=1)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--suppress-rdkit-warnings", dest="suppress_rdkit_warnings", action="store_true", default=True)
    parser.add_argument("--no-suppress-rdkit-warnings", dest="suppress_rdkit_warnings", action="store_false")
    parser.add_argument("--enable-camc", dest="enable_camc", action="store_true", default=True)
    parser.add_argument("--disable-camc", dest="enable_camc", action="store_false")
    parser.add_argument("--camc-delta-list", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--camc-top-k-list", nargs="+", type=int, default=None)
    parser.add_argument("--camc-min-motif-atoms", type=int, default=2)
    parser.add_argument("--camc-use-strict-theta", action="store_true", default=False)
    parser.add_argument("--camc-extraction-theta-list", nargs="+", type=float, default=None)
    parser.add_argument("--camc-reselect-ours-motifs", action="store_true", default=False)
    parser.add_argument(
        "--extra-fullgraph-selected-csv",
        action="append",
        default=[],
        help="Repeatable method_name:/path/to/selected_fullgraphs.csv for CAMC fullgraph baselines.",
    )
    return parser.parse_args()


def _coerce_label(value: Any) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    upper = text.upper()
    mapping = {
        "CI": 0,
        "CM": 1,
        "CA": 1,
        "INACTIVE": 0,
        "ACTIVE": 1,
        "FALSE": 0,
        "TRUE": 1,
        "NEGATIVE": 0,
        "POSITIVE": 1,
    }
    if upper in mapping:
        return mapping[upper]
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    integer = int(numeric)
    return integer if integer in (0, 1) else None


def _resolve_column(columns: Iterable[str], explicit: str | None, candidates: tuple[str, ...], kind: str) -> str:
    column_list = list(columns)
    if explicit:
        if explicit not in column_list:
            raise ValueError(f"Explicit {kind} column not found: {explicit}. Available={column_list}")
        return explicit
    for candidate in candidates:
        if candidate in column_list:
            return candidate
    raise ValueError(f"Could not infer {kind} column. Available={column_list}; candidates={candidates}")


def _canonicalize_smiles(smiles: str) -> tuple[str | None, int | None, int | None]:
    if Chem is None:
        raise RuntimeError("RDKit is required for this evaluator.")
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None, None
    return (
        Chem.MolToSmiles(mol, canonical=True),
        int(mol.GetNumAtoms()),
        int(mol.GetNumBonds()),
    )


def _sample_basic_records(
    records: list[BasicMoleculeRecord],
    max_count: int | None,
    seed: int,
) -> list[BasicMoleculeRecord]:
    if max_count is None or max_count <= 0 or len(records) <= max_count:
        return records
    rng = random.Random(seed)
    selected_positions = sorted(rng.sample(range(len(records)), max_count))
    return [records[position] for position in selected_positions]


def _score_basic_records(
    records: list[BasicMoleculeRecord],
    *,
    teacher: RFTeacher,
    target_label: int,
    disable_tqdm: bool,
    desc: str,
) -> list[MoleculeRecord]:
    scored: list[MoleculeRecord] = []
    iterator = progress_iter(records, desc=desc, total=len(records), disable_tqdm=disable_tqdm)
    for record in iterator:
        score = teacher.score(record.canonical_smiles, target_label=target_label)
        scored.append(
            MoleculeRecord(
                row_index=record.row_index,
                molecule_id=record.molecule_id,
                smiles=record.smiles,
                canonical_smiles=record.canonical_smiles,
                label=record.label,
                atom_count=record.atom_count,
                bond_count=record.bond_count,
                p_target=score.p_target,
                teacher_label=score.teacher_label,
                teacher_ok=score.ok,
                teacher_reason=score.reason,
            )
        )
    return scored


def load_hiv_rows(
    csv_path: Path,
    *,
    smiles_col: str | None,
    label_col: str | None,
) -> tuple[list[dict[str, str]], str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"HIV CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"HIV CSV has no header: {csv_path}")
        resolved_smiles_col = _resolve_column(reader.fieldnames, smiles_col, SMILES_COLUMN_CANDIDATES, "SMILES")
        resolved_label_col = _resolve_column(reader.fieldnames, label_col, LABEL_COLUMN_CANDIDATES, "label")
        raw_rows = list(reader)
    return raw_rows, resolved_smiles_col, resolved_label_col


def parse_hiv_records(
    raw_rows: list[dict[str, str]],
    *,
    resolved_smiles_col: str,
    resolved_label_col: str,
    disable_tqdm: bool,
) -> tuple[list[BasicMoleculeRecord], dict[str, Any]]:
    valid_records: list[BasicMoleculeRecord] = []
    invalid_smiles_count = 0
    invalid_label_count = 0
    iterator = progress_iter(
        enumerate(raw_rows),
        desc="parse SMILES",
        total=len(raw_rows),
        disable_tqdm=disable_tqdm,
    )
    for row_index, row in iterator:
        raw_smiles = str(row.get(resolved_smiles_col, "")).strip()
        label = _coerce_label(row.get(resolved_label_col, ""))
        if label is None:
            invalid_label_count += 1
            continue
        canonical_smiles, atom_count, bond_count = _canonicalize_smiles(raw_smiles)
        if canonical_smiles is None or atom_count is None or bond_count is None:
            invalid_smiles_count += 1
            continue
        valid_records.append(
            BasicMoleculeRecord(
                row_index=row_index,
                molecule_id=str(row.get("id") or row.get("ID") or row_index),
                smiles=raw_smiles,
                canonical_smiles=canonical_smiles,
                label=label,
                atom_count=atom_count,
                bond_count=bond_count,
            )
        )
    metadata = {
        "raw_row_count": len(raw_rows),
        "valid_record_count": len(valid_records),
        "invalid_smiles_count": invalid_smiles_count,
        "invalid_label_count": invalid_label_count,
    }
    return valid_records, metadata


def prepare_target_inputs(
    valid_records: list[BasicMoleculeRecord],
    *,
    target_label: int,
    teacher: RFTeacher,
    max_inputs: int | None,
    seed: int,
    disable_tqdm: bool,
) -> list[MoleculeRecord]:
    target_basic = [record for record in valid_records if record.label == int(target_label)]
    target_basic = _sample_basic_records(target_basic, max_inputs, seed)
    return _score_basic_records(
        target_basic,
        teacher=teacher,
        target_label=target_label,
        disable_tqdm=disable_tqdm,
        desc="score target inputs",
    )


def prepare_opposite_candidates(
    valid_records: list[BasicMoleculeRecord],
    *,
    target_label: int,
    teacher: RFTeacher,
    max_gt_candidates: int,
    seed: int,
    disable_tqdm: bool,
) -> tuple[list[MoleculeRecord], int]:
    opposite_seen: set[str] = set()
    opposite_basic: list[BasicMoleculeRecord] = []
    for record in valid_records:
        if record.label == int(target_label):
            continue
        if record.canonical_smiles in opposite_seen:
            continue
        opposite_seen.add(record.canonical_smiles)
        opposite_basic.append(record)
    before_sampling_count = len(opposite_basic)
    opposite_basic = _sample_basic_records(opposite_basic, max_gt_candidates, seed + 17)
    opposite_records = _score_basic_records(
        opposite_basic,
        teacher=teacher,
        target_label=target_label,
        disable_tqdm=disable_tqdm,
        desc="score opposite-label candidates",
    )
    return opposite_records, before_sampling_count


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fragment_from_mapping(row: dict[str, Any]) -> str | None:
    for key in FRAGMENT_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_selected_fragments(selected_dir: Path) -> tuple[list[FragmentRecord], dict[str, str]]:
    selected_dir = selected_dir.expanduser().resolve()
    if not selected_dir.exists():
        raise FileNotFoundError(f"ours selected dir not found: {selected_dir}")

    discovered = {
        "selector_summary": str(selected_dir / "selector_summary.json")
        if (selected_dir / "selector_summary.json").exists()
        else "",
        "selected_json": str(selected_dir / "selected_subgraphs.json")
        if (selected_dir / "selected_subgraphs.json").exists()
        else "",
        "selected_csv": str(selected_dir / "selected_subgraphs.csv")
        if (selected_dir / "selected_subgraphs.csv").exists()
        else "",
        "selector_report": str(selected_dir / "selector_report.txt")
        if (selected_dir / "selector_report.txt").exists()
        else "",
    }

    fragments: list[FragmentRecord] = []
    seen: set[str] = set()

    json_path = selected_dir / "selected_subgraphs.json"
    if json_path.exists():
        payload = _json_load(json_path)
        if isinstance(payload, list):
            for index, row in enumerate(payload, start=1):
                if not isinstance(row, dict):
                    continue
                fragment = _fragment_from_mapping(row)
                if fragment and fragment not in seen:
                    seen.add(fragment)
                    fragments.append(FragmentRecord(int(row.get("rank") or index), fragment, str(json_path)))

    csv_path = selected_dir / "selected_subgraphs.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                fragment = _fragment_from_mapping(row)
                if fragment and fragment not in seen:
                    seen.add(fragment)
                    rank = int(float(row.get("rank") or index))
                    fragments.append(FragmentRecord(rank, fragment, str(csv_path)))

    summary_path = selected_dir / "selector_summary.json"
    if summary_path.exists():
        payload = _json_load(summary_path)
        if isinstance(payload, dict):
            selected = payload.get("selected_fragments")
            if isinstance(selected, list):
                for index, fragment_value in enumerate(selected, start=1):
                    fragment = str(fragment_value).strip()
                    if fragment and fragment not in seen:
                        seen.add(fragment)
                        fragments.append(FragmentRecord(index, fragment, str(summary_path)))

    report_path = selected_dir / "selector_report.txt"
    if report_path.exists():
        pattern = re.compile(r"\bfragment=([^\s]+)")
        for index, line in enumerate(report_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            match = pattern.search(line)
            if not match:
                continue
            fragment = match.group(1).strip()
            if fragment and fragment not in seen:
                seen.add(fragment)
                fragments.append(FragmentRecord(index, fragment, str(report_path)))

    fragments.sort(key=lambda item: (item.rank, item.fragment))
    if not fragments:
        raise ValueError(
            f"No selected fragments could be parsed from {selected_dir}. "
            f"Discovered files: {discovered}"
        )
    return fragments, discovered


def _base_ours_action_row(
    *,
    record: MoleculeRecord,
    input_idx: int,
    fragment: FragmentRecord,
    target_label: int,
) -> dict[str, Any]:
    return {
        "method": "ours_selected_subgraph",
        "target_label": int(target_label),
        "input_idx": int(input_idx),
        "input_row_index": record.row_index,
        "input_id": record.molecule_id,
        "parent_smiles": record.canonical_smiles,
        "parent_label": record.label,
        "fragment_rank": int(fragment.rank),
        "fragment": fragment.fragment,
        "match_ok": False,
        "delete_ok": False,
        "valid_after": False,
        "recourse_smiles": None,
        "distance_proxy": None,
        "proxy_edit": None,
        "p_before": record.p_target,
        "p_after": None,
        "before_teacher_label": record.teacher_label,
        "after_teacher_label": None,
        "cf_drop": None,
        "cf_flip": False,
        "failure_reason": "",
    }


def evaluate_ours_action_candidates(
    records: list[MoleculeRecord],
    fragments: list[FragmentRecord],
    *,
    top_k_max: int,
    target_label: int,
    teacher: RFTeacher,
    distance_proxy: DistanceProxy,
    progress_every: int,
    disable_tqdm: bool,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_fragments = fragments[: min(top_k_max, len(fragments))]
    rows: list[dict[str, Any]] = []
    start = time.time()
    processed_inputs = 0
    match_ok_count = 0
    delete_ok_count = 0
    valid_after_count = 0
    any_flip_count = 0

    iterator = progress_iter(
        enumerate(records),
        desc="evaluate ours action candidates",
        total=len(records),
        disable_tqdm=disable_tqdm,
    )
    for input_idx, record in iterator:
        processed_inputs += 1
        input_match = False
        input_delete = False
        input_valid = False
        input_flip = False
        for fragment in active_fragments:
            row = _base_ours_action_row(record=record, input_idx=input_idx, fragment=fragment, target_label=target_label)
            if not record.teacher_ok or record.p_target is None:
                row["failure_reason"] = f"before_teacher_failed:{record.teacher_reason}"
                rows.append(row)
                continue

            try:
                match_ok = is_parent_substructure(record.canonical_smiles, fragment.fragment)
            except Exception as exc:
                row["failure_reason"] = f"substructure_check_failed:{exc}"
                rows.append(row)
                continue
            row["match_ok"] = bool(match_ok)
            if not match_ok:
                row["failure_reason"] = "no_selected_fragment_substructure"
                rows.append(row)
                continue
            input_match = True

            deletion = delete_fragment_from_parent(record.canonical_smiles, fragment.fragment)
            if not deletion.success or deletion.residual_smiles is None:
                row["failure_reason"] = deletion.failure_reason or "fragment_deletion_failed"
                rows.append(row)
                continue
            row["delete_ok"] = True
            input_delete = True

            residual_smiles = prepare_smiles_for_oracle(deletion.residual_smiles)
            if residual_smiles is None or residual_smiles == "":
                row["failure_reason"] = "residual_prepare_failed"
                rows.append(row)
                continue
            row["recourse_smiles"] = residual_smiles

            score_after = teacher.score(residual_smiles, target_label=target_label)
            if not score_after.ok or score_after.p_target is None:
                row["failure_reason"] = f"after_teacher_failed:{score_after.reason}"
                rows.append(row)
                continue
            row["valid_after"] = True
            row["p_after"] = float(score_after.p_target)
            row["after_teacher_label"] = score_after.teacher_label
            row["cf_drop"] = float(record.p_target) - float(score_after.p_target)
            row["cf_flip"] = teacher_strict_flip(
                record.teacher_label,
                score_after.teacher_label,
                target_label,
            )
            input_valid = True
            if row["cf_flip"]:
                input_flip = True

            distance = distance_proxy.distance(record.canonical_smiles, residual_smiles)
            if not distance["ok"] or distance["cost"] is None:
                row["failure_reason"] = distance["reason"]
                rows.append(row)
                continue
            row["distance_proxy"] = float(distance["cost"])
            row["proxy_edit"] = float(distance["proxy_edit"])
            row["failure_reason"] = "ok"
            rows.append(row)

        match_ok_count += int(input_match)
        delete_ok_count += int(input_delete)
        valid_after_count += int(input_valid)
        any_flip_count += int(input_flip)
        if progress_every > 0 and processed_inputs % progress_every == 0:
            elapsed = time.time() - start
            logger.info(
                "ours_progress processed_inputs=%d match_ok_count=%d delete_ok_count=%d "
                "valid_after_count=%d any_flip_count=%d elapsed_sec=%.3f avg_sec_per_input=%.6f",
                processed_inputs,
                match_ok_count,
                delete_ok_count,
                valid_after_count,
                any_flip_count,
                elapsed,
                elapsed / float(processed_inputs),
            )

    elapsed = time.time() - start
    logger.info(
        "ours_progress processed_inputs=%d match_ok_count=%d delete_ok_count=%d "
        "valid_after_count=%d any_flip_count=%d elapsed_sec=%.3f avg_sec_per_input=%.6f",
        processed_inputs,
        match_ok_count,
        delete_ok_count,
        valid_after_count,
        any_flip_count,
        elapsed,
        elapsed / float(processed_inputs) if processed_inputs else 0.0,
    )
    return rows, {
        "processed_inputs": processed_inputs,
        "match_ok_count": match_ok_count,
        "delete_ok_count": delete_ok_count,
        "valid_after_count": valid_after_count,
        "any_flip_count": any_flip_count,
        "elapsed_sec": elapsed,
        "avg_sec_per_input": elapsed / float(processed_inputs) if processed_inputs else 0.0,
    }


def _actions_by_input(action_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in action_rows:
        grouped.setdefault(int(row["input_idx"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["fragment_rank"]))
    return grouped


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_from_action_aggregate(
    *,
    method: str,
    target_label: int,
    k: int,
    theta: float,
    input_idx: int,
    record: MoleculeRecord,
    active_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_after_actions = [
        row
        for row in active_actions
        if row.get("match_ok") and row.get("delete_ok") and row.get("valid_after")
    ]
    flip_actions = [row for row in valid_after_actions if bool(row.get("cf_flip"))]
    feasible_actions = [
        row
        for row in flip_actions
        if _float_or_none(row.get("distance_proxy")) is not None
        and float(row["distance_proxy"]) <= float(theta)
    ]
    representative: dict[str, Any] | None = None
    if feasible_actions:
        representative = max(
            feasible_actions,
            key=lambda row: (float(row.get("cf_drop") or -1e9), -float(row.get("distance_proxy") or 1e9)),
        )
    elif flip_actions:
        representative = max(
            flip_actions,
            key=lambda row: (float(row.get("cf_drop") or -1e9), -float(row.get("distance_proxy") or 1e9)),
        )
    elif valid_after_actions:
        representative = max(valid_after_actions, key=lambda row: float(row.get("cf_drop") or -1e9))

    covered = bool(feasible_actions)
    cost = min(float(row["distance_proxy"]) for row in feasible_actions) if feasible_actions else None
    proxy_edit = min(float(row["proxy_edit"]) for row in feasible_actions if row.get("proxy_edit") is not None) if feasible_actions else None
    if feasible_actions:
        cf_drop = max(float(row["cf_drop"]) for row in feasible_actions if row.get("cf_drop") is not None)
    elif valid_after_actions:
        cf_drop = max(float(row["cf_drop"]) for row in valid_after_actions if row.get("cf_drop") is not None)
    else:
        cf_drop = None

    reason = "ok" if covered else "no_theta_feasible_flip"
    if representative is None:
        failure_reasons = [str(row.get("failure_reason") or "") for row in active_actions if row.get("failure_reason")]
        reason = failure_reasons[0] if failure_reasons else "no_candidate_action"

    return {
        "method": method,
        "target_label": int(target_label),
        "k": int(k),
        "theta": float(theta),
        "input_idx": int(input_idx),
        "input_row_index": record.row_index,
        "input_id": record.molecule_id,
        "parent_smiles": record.canonical_smiles,
        "parent_label": record.label,
        "p_before": record.p_target,
        "before_teacher_label": record.teacher_label,
        "recourse_smiles": representative.get("recourse_smiles") if representative else None,
        "fragment_rank": representative.get("fragment_rank") if representative else None,
        "fragment_smiles": representative.get("fragment") if representative else None,
        "candidate_rank": None,
        "candidate_index": None,
        "candidate_label": None,
        "p_after": representative.get("p_after") if representative else None,
        "after_teacher_label": representative.get("after_teacher_label") if representative else None,
        "cf_drop": cf_drop,
        "cf_flip": bool(flip_actions),
        "cost": cost,
        "proxy_edit": proxy_edit,
        "valid_recourse": bool(valid_after_actions),
        "covered": covered,
        "coverage_theta": covered,
        "coverage_unconstrained_flip": bool(flip_actions),
        "reason": reason,
    }


def aggregate_ours_recourse_rows(
    records: list[MoleculeRecord],
    action_rows: list[dict[str, Any]],
    *,
    top_k_list: list[int],
    theta_list: list[float],
    target_label: int,
) -> list[dict[str, Any]]:
    grouped = _actions_by_input(action_rows)
    rows: list[dict[str, Any]] = []
    for k in top_k_list:
        for theta in theta_list:
            for input_idx, record in enumerate(records):
                active = [row for row in grouped.get(input_idx, []) if int(row["fragment_rank"]) <= int(k)]
                rows.append(
                    _row_from_action_aggregate(
                        method="ours_selected_subgraph",
                        target_label=target_label,
                        k=k,
                        theta=theta,
                        input_idx=input_idx,
                        record=record,
                        active_actions=active,
                    )
                )
    return rows


def build_ours_diagnostics(
    action_rows: list[dict[str, Any]],
    *,
    records: list[MoleculeRecord],
    top_k_list: list[int],
    theta_list: list[float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    grouped = _actions_by_input(action_rows)
    num_inputs = len(records)
    per_k_match_rate: dict[str, float] = {}
    per_k_delete_ok_rate: dict[str, float] = {}
    per_k_any_flip_rate: dict[str, float] = {}
    per_k_any_feasible_by_theta: dict[str, dict[str, float]] = {}
    for k in top_k_list:
        match_count = 0
        delete_count = 0
        flip_count = 0
        theta_counts = {str(theta): 0 for theta in theta_list}
        for input_idx in range(num_inputs):
            active = [row for row in grouped.get(input_idx, []) if int(row["fragment_rank"]) <= int(k)]
            if any(row.get("match_ok") for row in active):
                match_count += 1
            if any(row.get("delete_ok") for row in active):
                delete_count += 1
            flip_actions = [
                row
                for row in active
                if row.get("match_ok")
                and row.get("delete_ok")
                and row.get("valid_after")
                and row.get("cf_flip")
            ]
            if flip_actions:
                flip_count += 1
            for theta in theta_list:
                if any(
                    _float_or_none(row.get("distance_proxy")) is not None
                    and float(row["distance_proxy"]) <= float(theta)
                    for row in flip_actions
                ):
                    theta_counts[str(theta)] += 1
        per_k_match_rate[str(k)] = float(match_count) / float(num_inputs) if num_inputs else 0.0
        per_k_delete_ok_rate[str(k)] = float(delete_count) / float(num_inputs) if num_inputs else 0.0
        per_k_any_flip_rate[str(k)] = float(flip_count) / float(num_inputs) if num_inputs else 0.0
        per_k_any_feasible_by_theta[str(k)] = {
            theta: float(count) / float(num_inputs) if num_inputs else 0.0
            for theta, count in theta_counts.items()
        }
    return per_k_match_rate, per_k_delete_ok_rate, per_k_any_flip_rate, per_k_any_feasible_by_theta


def build_gt_candidate_matrix(
    records: list[MoleculeRecord],
    candidates: list[MoleculeRecord],
    *,
    target_label: int,
    distance_proxy: DistanceProxy,
    disable_tqdm: bool,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[int]]:
    matrix: dict[tuple[int, int], dict[str, Any]] = {}
    eligible_candidate_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.teacher_ok and candidate.p_target is not None and candidate.teacher_label != int(target_label)
    ]
    iterator = progress_iter(
        eligible_candidate_indices,
        desc="GT pairwise MCS precompute",
        total=len(eligible_candidate_indices),
        disable_tqdm=disable_tqdm,
    )
    for candidate_index in iterator:
        candidate = candidates[candidate_index]
        for input_idx, record in enumerate(records):
            if not record.teacher_ok or record.p_target is None:
                continue
            distance = distance_proxy.distance(record.canonical_smiles, candidate.canonical_smiles)
            if not distance["ok"] or distance["cost"] is None:
                continue
            matrix[(input_idx, candidate_index)] = {
                "cost": float(distance["cost"]),
                "proxy_edit": float(distance["proxy_edit"]),
                "cf_drop": float(record.p_target) - float(candidate.p_target),
                "cf_flip": teacher_strict_flip(
                    record.teacher_label,
                    candidate.teacher_label,
                    target_label,
                ),
            }
    return matrix, eligible_candidate_indices


def greedy_select_gt_fullgraphs(
    records: list[MoleculeRecord],
    candidates: list[MoleculeRecord],
    *,
    candidate_matrix: dict[tuple[int, int], dict[str, Any]],
    eligible_candidate_indices: list[int],
    theta_list: list[float],
    target_label: int,
    top_k_max: int,
    disable_tqdm: bool,
) -> tuple[list[int], list[dict[str, Any]]]:
    selection_theta = max(theta_list) if theta_list else 0.0
    coverage_at_selection: dict[int, set[int]] = {}
    for candidate_index in eligible_candidate_indices:
        coverage_at_selection[candidate_index] = {
            input_idx
            for input_idx in range(len(records))
            if (input_idx, candidate_index) in candidate_matrix
            and bool(candidate_matrix[(input_idx, candidate_index)].get("cf_flip"))
            and float(candidate_matrix[(input_idx, candidate_index)]["cost"]) <= float(selection_theta)
        }

    selected: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    covered: set[int] = set()
    remaining = set(eligible_candidate_indices)
    rounds = range(min(top_k_max, len(remaining)))
    iterator = progress_iter(rounds, desc="greedy select GT fullgraphs", total=len(list(rounds)), disable_tqdm=disable_tqdm)
    for _ in iterator:
        best_index: int | None = None
        best_key: tuple[float, float, float, float] | None = None
        for candidate_index in remaining:
            new_covered = coverage_at_selection[candidate_index] - covered
            gain = len(new_covered)
            if gain > 0:
                cf_gain = sum(
                    float(candidate_matrix[(input_idx, candidate_index)]["cf_drop"])
                    for input_idx in new_covered
                    if (input_idx, candidate_index) in candidate_matrix
                )
                cost_gain = sum(
                    float(candidate_matrix[(input_idx, candidate_index)]["cost"])
                    for input_idx in new_covered
                    if (input_idx, candidate_index) in candidate_matrix
                )
                mean_cost_gain = cost_gain / gain
            else:
                cf_gain = float(candidates[candidate_index].p_target or 0.0) * -1.0
                mean_cost_gain = 1e9
            key = (float(gain), float(cf_gain), -float(mean_cost_gain), -float(candidate_index))
            if best_key is None or key > best_key:
                best_key = key
                best_index = candidate_index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        covered.update(coverage_at_selection[best_index])

        rank = len(selected)
        for theta in theta_list:
            theta_covered = {
                input_idx
                for input_idx in range(len(records))
                for selected_index in selected
                if (input_idx, selected_index) in candidate_matrix
                and bool(candidate_matrix[(input_idx, selected_index)].get("cf_flip"))
                and float(candidate_matrix[(input_idx, selected_index)]["cost"]) <= float(theta)
            }
            previous_prefix = selected[:-1]
            prev_covered = {
                input_idx
                for input_idx in range(len(records))
                for selected_index in previous_prefix
                if (input_idx, selected_index) in candidate_matrix
                and bool(candidate_matrix[(input_idx, selected_index)].get("cf_flip"))
                and float(candidate_matrix[(input_idx, selected_index)]["cost"]) <= float(theta)
            }
            new_covered = theta_covered - prev_covered
            candidate = candidates[best_index]
            selected_rows.append(
                {
                    "method": "gt_fullgraph_greedy",
                    "target_label": int(target_label),
                    "theta": float(theta),
                    "selection_theta": float(selection_theta),
                    "rank": rank,
                    "candidate_index": best_index,
                    "candidate_row_index": candidate.row_index,
                    "candidate_id": candidate.molecule_id,
                    "candidate_smiles": candidate.canonical_smiles,
                    "candidate_label": candidate.label,
                    "candidate_p_target": candidate.p_target,
                    "candidate_teacher_label": candidate.teacher_label,
                    "coverage_gain_count": len(new_covered),
                    "coverage_gain": float(len(new_covered)) / float(len(records)) if records else 0.0,
                    "cumulative_covered_count": len(theta_covered),
                    "cumulative_coverage": float(len(theta_covered)) / float(len(records)) if records else 0.0,
                    "covered_input_indices": "|".join(str(index) for index in sorted(new_covered)[:50]),
                }
            )
    return selected, selected_rows


def aggregate_gt_recourse_rows(
    records: list[MoleculeRecord],
    candidates: list[MoleculeRecord],
    *,
    selected_indices: list[int],
    candidate_matrix: dict[tuple[int, int], dict[str, Any]],
    top_k_list: list[int],
    theta_list: list[float],
    target_label: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_rank = {candidate_index: rank for rank, candidate_index in enumerate(selected_indices, start=1)}
    for k in top_k_list:
        prefix = selected_indices[: min(k, len(selected_indices))]
        for theta in theta_list:
            for input_idx, record in enumerate(records):
                valid_entries = [
                    (candidate_index, candidate_matrix[(input_idx, candidate_index)])
                    for candidate_index in prefix
                    if (input_idx, candidate_index) in candidate_matrix
                ]
                feasible_entries = [
                    (candidate_index, entry)
                    for candidate_index, entry in valid_entries
                    if float(entry["cost"]) <= float(theta) and bool(entry.get("cf_flip"))
                ]
                covered = bool(feasible_entries)
                representative: tuple[int, dict[str, Any]] | None = None
                if feasible_entries:
                    representative = max(feasible_entries, key=lambda item: (float(item[1]["cf_drop"]), -float(item[1]["cost"])))
                elif valid_entries:
                    representative = min(valid_entries, key=lambda item: float(item[1]["cost"]))

                cost = min(float(entry["cost"]) for _, entry in feasible_entries) if feasible_entries else None
                proxy_edit = min(float(entry["proxy_edit"]) for _, entry in feasible_entries) if feasible_entries else None
                cf_drop = max(float(entry["cf_drop"]) for _, entry in feasible_entries) if feasible_entries else None
                candidate_index: int | None = representative[0] if representative else None
                entry: dict[str, Any] | None = representative[1] if representative else None
                candidate = candidates[candidate_index] if candidate_index is not None else None
                rows.append(
                    {
                        "method": "gt_fullgraph_greedy",
                        "target_label": int(target_label),
                        "k": int(k),
                        "theta": float(theta),
                        "input_idx": int(input_idx),
                        "input_row_index": record.row_index,
                        "input_id": record.molecule_id,
                        "parent_smiles": record.canonical_smiles,
                        "parent_label": record.label,
                        "p_before": record.p_target,
                        "before_teacher_label": record.teacher_label,
                        "recourse_smiles": candidate.canonical_smiles if candidate else None,
                        "fragment_rank": None,
                        "fragment_smiles": None,
                        "candidate_rank": selected_rank.get(candidate_index) if candidate_index is not None else None,
                        "candidate_index": candidate_index,
                        "candidate_label": candidate.label if candidate else None,
                        "p_after": candidate.p_target if candidate else None,
                        "after_teacher_label": candidate.teacher_label if candidate else None,
                        "cf_drop": cf_drop if cf_drop is not None else (entry.get("cf_drop") if entry else None),
                        "cf_flip": any(bool(item.get("cf_flip")) for _, item in valid_entries),
                        "cost": cost,
                        "proxy_edit": proxy_edit,
                        "valid_recourse": bool(valid_entries),
                        "covered": covered,
                        "coverage_theta": covered,
                        "coverage_unconstrained_flip": any(
                            bool(item.get("cf_flip")) for _, item in valid_entries
                        ),
                        "reason": "ok" if covered else ("no_theta_feasible_fullgraph" if valid_entries else "no_selected_candidate"),
                    }
                )
    return rows


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary_for_rows(
    rows: list[dict[str, Any]],
    *,
    method: str,
    target_label: int,
    k: int,
    theta: float,
    num_inputs: int,
    runtime_sec: float,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_recourse")]
    covered = [row for row in rows if row.get("covered")]
    covered_costs = [float(row["cost"]) for row in covered if row.get("cost") not in (None, "")]
    valid_cf_drops = [float(row["cf_drop"]) for row in valid if row.get("cf_drop") not in (None, "")]
    covered_cf_drops = [float(row["cf_drop"]) for row in covered if row.get("cf_drop") not in (None, "")]
    unconstrained = [row for row in rows if row.get("coverage_unconstrained_flip")]
    coverage = float(len(covered)) / float(num_inputs) if num_inputs else 0.0
    unconstrained_flip = float(len(unconstrained)) / float(num_inputs) if num_inputs else 0.0
    return {
        "method": method,
        "target_label": int(target_label),
        "k": int(k),
        "theta": float(theta),
        "num_inputs": int(num_inputs),
        "coverage": coverage,
        "coverage_theta": coverage,
        "coverage_unconstrained_flip": unconstrained_flip,
        "median_cost": float(median(covered_costs)) if covered_costs else None,
        "mean_cost": _safe_mean(covered_costs),
        "median_cost_covered_only": float(median(covered_costs)) if covered_costs else None,
        "mean_cf_drop": _safe_mean(valid_cf_drops),
        "mean_cf_drop_covered_only": _safe_mean(covered_cf_drops),
        "flip_rate": unconstrained_flip,
        "valid_recourse_rate": float(len(valid)) / float(num_inputs) if num_inputs else 0.0,
        "runtime_sec": float(runtime_sec),
    }


def build_comparison_table(
    rows_by_method: dict[str, list[dict[str, Any]]],
    *,
    target_label: int,
    top_k_list: list[int],
    theta_list: list[float],
    num_inputs: int,
    runtime_sec: float,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for method, method_rows in rows_by_method.items():
        for k in top_k_list:
            for theta in theta_list:
                subset = [
                    row
                    for row in method_rows
                    if int(row["k"]) == int(k) and abs(float(row["theta"]) - float(theta)) < 1e-12
                ]
                table.append(
                    _summary_for_rows(
                        subset,
                        method=method,
                        target_label=target_label,
                        k=k,
                        theta=theta,
                        num_inputs=num_inputs,
                        runtime_sec=runtime_sec,
                    )
                )
    return table


def check_recourse_monotonicity(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in table})
    eps = 1e-12
    for method in methods:
        method_rows = [row for row in table if str(row["method"]) == method]
        theta_values = sorted({float(row["theta"]) for row in method_rows})
        k_values = sorted({int(row["k"]) for row in method_rows})
        for theta in theta_values:
            prev_row: dict[str, Any] | None = None
            for k in k_values:
                row = next((item for item in method_rows if int(item["k"]) == k and abs(float(item["theta"]) - theta) < eps), None)
                if row is None:
                    continue
                if prev_row is not None and float(row["coverage"]) + eps < float(prev_row["coverage"]):
                    warnings.append(
                        {
                            "method": method,
                            "metric": "recourse_coverage",
                            "theta": theta,
                            "k_prev": int(prev_row["k"]),
                            "k_next": int(row["k"]),
                            "value_prev": float(prev_row["coverage"]),
                            "value_next": float(row["coverage"]),
                            "warning_type": "coverage_decreased_with_k",
                        }
                    )
                prev_row = row
        for k in k_values:
            prev_row = None
            for theta in theta_values:
                row = next((item for item in method_rows if int(item["k"]) == k and abs(float(item["theta"]) - theta) < eps), None)
                if row is None:
                    continue
                if prev_row is not None and float(row["coverage"]) + eps < float(prev_row["coverage"]):
                    warnings.append(
                        {
                            "method": method,
                            "metric": "recourse_coverage",
                            "k": int(k),
                            "theta_prev": float(prev_row["theta"]),
                            "theta_next": float(row["theta"]),
                            "value_prev": float(prev_row["coverage"]),
                            "value_next": float(row["coverage"]),
                            "warning_type": "coverage_decreased_with_theta",
                        }
                    )
                prev_row = row
    return warnings


def safe_method_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method).strip("_") or "method"


def canonicalize_motif_smiles(smiles: str, *, min_atoms: int) -> dict[str, Any]:
    canonical, atom_count, bond_count = _canonicalize_smiles(smiles)
    if canonical is None or atom_count is None or bond_count is None:
        return {
            "valid_motif": False,
            "canonical_motif_smiles": None,
            "atom_count": None,
            "bond_count": None,
            "failure_reason": "motif_parse_failed",
        }
    valid = int(atom_count) >= int(min_atoms)
    return {
        "valid_motif": bool(valid),
        "canonical_motif_smiles": canonical,
        "atom_count": int(atom_count),
        "bond_count": int(bond_count),
        "failure_reason": "ok" if valid else "motif_too_small",
    }


def build_ours_action_motifs(
    fragments: list[FragmentRecord],
    *,
    min_atoms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, fragment in enumerate(fragments, start=1):
        info = canonicalize_motif_smiles(fragment.fragment, min_atoms=min_atoms)
        rows.append(
            {
                "method": "ours_selected_subgraph",
                "action_source": "selected_fragment",
                "motif_rank": index,
                "motif_smiles": fragment.fragment,
                "source_fragment_rank": fragment.rank,
                "source": fragment.source,
                "atom_count": info["atom_count"],
                "bond_count": info["bond_count"],
                "valid_motif": info["valid_motif"],
                "canonical_motif_smiles": info["canonical_motif_smiles"],
                "failure_reason": info["failure_reason"],
            }
        )
    return rows


def molecule_record_to_fullgraph(
    record: MoleculeRecord,
    *,
    method: str,
    rank: int,
    source_index: int,
    source_path: str,
) -> FullgraphRecord:
    return FullgraphRecord(
        method=method,
        rank=int(rank),
        source_index=int(source_index),
        source_id=record.molecule_id,
        source_path=source_path,
        smiles=record.smiles,
        canonical_smiles=record.canonical_smiles,
        label=record.label,
        atom_count=record.atom_count,
        bond_count=record.bond_count,
        p_target=record.p_target,
        teacher_label=record.teacher_label,
        teacher_ok=record.teacher_ok,
        teacher_reason=record.teacher_reason,
    )


def parse_extra_fullgraph_spec(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(
            "--extra-fullgraph-selected-csv must use method_name:/path/to/selected_fullgraphs.csv"
        )
    method, path_text = spec.split(":", 1)
    method = method.strip()
    if not method:
        raise ValueError("--extra-fullgraph-selected-csv method_name cannot be empty.")
    path = Path(path_text).expanduser()
    return method, path


def load_extra_fullgraph_selected_csv(
    spec: str,
    *,
    teacher: RFTeacher,
    target_label: int,
    disable_tqdm: bool,
) -> list[FullgraphRecord]:
    method, path = parse_extra_fullgraph_spec(spec)
    if not path.exists():
        raise FileNotFoundError(f"extra fullgraph selected CSV not found for {method}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"extra fullgraph selected CSV has no header for {method}: {path}")
        smiles_col = None
        for candidate in FULLGRAPH_SMILES_COLUMNS:
            if candidate in reader.fieldnames:
                smiles_col = candidate
                break
        if smiles_col is None:
            raise ValueError(
                "Graph benchmark output is not supported in HIV/SMILES CAMC evaluator. "
                "Please provide selected fullgraphs as SMILES or use a graph-level CAMC evaluator."
            )
        raw_rows = list(reader)

    records: list[FullgraphRecord] = []
    seen: set[str] = set()
    iterator = progress_iter(
        enumerate(raw_rows, start=1),
        desc=f"load extra fullgraph {method}",
        total=len(raw_rows),
        disable_tqdm=disable_tqdm,
    )
    for index, row in iterator:
        raw_smiles = str(row.get(smiles_col, "")).strip()
        if not raw_smiles:
            continue
        canonical, atom_count, bond_count = _canonicalize_smiles(raw_smiles)
        if canonical is None or atom_count is None or bond_count is None:
            raise ValueError(f"Invalid SMILES in extra fullgraph CSV {path} row {index}: {raw_smiles}")
        if canonical in seen:
            continue
        seen.add(canonical)
        score = teacher.score(canonical, target_label=target_label)
        rank = int(float(row.get("rank") or index))
        label = _coerce_label(row.get("label", "")) if "label" in row else None
        records.append(
            FullgraphRecord(
                method=method,
                rank=rank,
                source_index=index,
                source_id=str(row.get("id") or row.get("ID") or index),
                source_path=str(path),
                smiles=raw_smiles,
                canonical_smiles=canonical,
                label=label,
                atom_count=int(atom_count),
                bond_count=int(bond_count),
                p_target=score.p_target,
                teacher_label=score.teacher_label,
                teacher_ok=score.ok,
                teacher_reason=score.reason,
                score=_float_or_none(row.get("score")),
            )
        )
    if not records:
        raise ValueError(f"No parseable fullgraph SMILES found in {path} for method {method}.")
    records.sort(key=lambda item: (item.rank, item.canonical_smiles))
    return records


def _deleted_components(parent_mol: Any, deleted_atoms: set[int]) -> list[set[int]]:
    remaining = set(deleted_atoms)
    components: list[set[int]] = []
    adjacency: dict[int, set[int]] = {atom_idx: set() for atom_idx in deleted_atoms}
    for bond in parent_mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in deleted_atoms and end in deleted_atoms:
            adjacency[begin].add(end)
            adjacency[end].add(begin)
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        remaining.remove(root)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda comp: (-len(comp), sorted(comp)))
    return components


def extract_deleted_motif_from_fullgraph(
    parent_smiles: str,
    fullgraph_smiles: str,
    *,
    min_atoms: int,
    distance_proxy: DistanceProxy,
) -> dict[str, Any]:
    parent_mol = distance_proxy.mol(parent_smiles)
    fullgraph_mol = distance_proxy.mol(fullgraph_smiles)
    if parent_mol is None or fullgraph_mol is None:
        return {"valid_motif": False, "failure_reason": "parent_or_fullgraph_parse_failed"}
    mcs = distance_proxy.find_mcs(parent_smiles, fullgraph_smiles)
    if not mcs.ok or not mcs.smarts:
        return {
            "valid_motif": False,
            "failure_reason": mcs.reason,
            "mcs_num_atoms": mcs.mcs_atoms,
            "mcs_num_bonds": mcs.mcs_bonds,
        }
    query = Chem.MolFromSmarts(mcs.smarts)
    if query is None:
        return {
            "valid_motif": False,
            "failure_reason": "mcs_query_parse_failed",
            "mcs_num_atoms": mcs.mcs_atoms,
            "mcs_num_bonds": mcs.mcs_bonds,
        }
    parent_match = parent_mol.GetSubstructMatch(query)
    if not parent_match:
        return {
            "valid_motif": False,
            "failure_reason": "mcs_parent_match_failed",
            "mcs_num_atoms": mcs.mcs_atoms,
            "mcs_num_bonds": mcs.mcs_bonds,
        }
    all_parent_atoms = set(range(int(parent_mol.GetNumAtoms())))
    deleted_atoms = all_parent_atoms - set(int(index) for index in parent_match)
    if not deleted_atoms:
        return {
            "valid_motif": False,
            "failure_reason": "deleted_motif_empty",
            "mcs_num_atoms": mcs.mcs_atoms,
            "mcs_num_bonds": mcs.mcs_bonds,
            "component_count": 0,
        }
    components = _deleted_components(parent_mol, deleted_atoms)
    selected_component = components[0]
    try:
        motif_smiles = Chem.MolFragmentToSmiles(
            parent_mol,
            atomsToUse=sorted(selected_component),
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        return {
            "valid_motif": False,
            "failure_reason": f"deleted_motif_smiles_failed:{exc}",
            "mcs_num_atoms": mcs.mcs_atoms,
            "mcs_num_bonds": mcs.mcs_bonds,
            "component_count": len(components),
        }
    info = canonicalize_motif_smiles(motif_smiles, min_atoms=min_atoms)
    return {
        "deleted_motif_smiles": motif_smiles,
        "canonical_motif_smiles": info["canonical_motif_smiles"],
        "motif_atom_count": info["atom_count"],
        "motif_bond_count": info["bond_count"],
        "valid_motif": info["valid_motif"],
        "failure_reason": info["failure_reason"],
        "mcs_num_atoms": mcs.mcs_atoms,
        "mcs_num_bonds": mcs.mcs_bonds,
        "component_count": len(components),
    }


def extract_fullgraph_action_motif_pool(
    *,
    method: str,
    records: list[MoleculeRecord],
    selected_fullgraphs: list[FullgraphRecord],
    target_label: int,
    min_atoms: int,
    use_strict_theta: bool,
    extraction_theta_list: list[float],
    distance_proxy: DistanceProxy,
    disable_tqdm: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    theta_limit = max(extraction_theta_list) if extraction_theta_list else None
    iterator = progress_iter(
        enumerate(records),
        desc=f"extract {method} action motifs",
        total=len(records),
        disable_tqdm=disable_tqdm,
    )
    for input_idx, record in iterator:
        nearest: tuple[float, int, FullgraphRecord, dict[str, Any]] | None = None
        for fullgraph in selected_fullgraphs:
            if not fullgraph.teacher_ok or fullgraph.teacher_label == int(target_label):
                continue
            distance = distance_proxy.distance(record.canonical_smiles, fullgraph.canonical_smiles)
            if not distance["ok"] or distance["cost"] is None:
                continue
            cost = float(distance["cost"])
            if use_strict_theta and theta_limit is not None and cost > float(theta_limit):
                continue
            key = (cost, int(fullgraph.rank))
            if nearest is None or key < (nearest[0], nearest[1]):
                nearest = (cost, int(fullgraph.rank), fullgraph, distance)

        if nearest is None:
            rows.append(
                {
                    "method": method,
                    "input_idx": input_idx,
                    "parent_smiles": record.canonical_smiles,
                    "nearest_fullgraph_rank": None,
                    "nearest_fullgraph_smiles": None,
                    "nearest_distance_proxy": None,
                    "fullgraph_cf_flip": False,
                    "mcs_num_atoms": None,
                    "mcs_num_bonds": None,
                    "deleted_motif_smiles": None,
                    "canonical_motif_smiles": None,
                    "motif_atom_count": None,
                    "motif_bond_count": None,
                    "component_count": None,
                    "valid_motif": False,
                    "failure_reason": "no_cf_fullgraph_candidate",
                }
            )
            continue

        cost, rank, fullgraph, _ = nearest
        motif = extract_deleted_motif_from_fullgraph(
            record.canonical_smiles,
            fullgraph.canonical_smiles,
            min_atoms=min_atoms,
            distance_proxy=distance_proxy,
        )
        rows.append(
            {
                "method": method,
                "input_idx": input_idx,
                "parent_smiles": record.canonical_smiles,
                "nearest_fullgraph_rank": rank,
                "nearest_fullgraph_smiles": fullgraph.canonical_smiles,
                "nearest_distance_proxy": cost,
                "fullgraph_cf_flip": teacher_strict_flip(
                    record.teacher_label,
                    fullgraph.teacher_label,
                    target_label,
                ),
                "mcs_num_atoms": motif.get("mcs_num_atoms"),
                "mcs_num_bonds": motif.get("mcs_num_bonds"),
                "deleted_motif_smiles": motif.get("deleted_motif_smiles"),
                "canonical_motif_smiles": motif.get("canonical_motif_smiles"),
                "motif_atom_count": motif.get("motif_atom_count"),
                "motif_bond_count": motif.get("motif_bond_count"),
                "component_count": motif.get("component_count"),
                "valid_motif": bool(motif.get("valid_motif")),
                "failure_reason": motif.get("failure_reason"),
            }
        )
    return rows


def dedupe_fullgraph_motif_pool(method: str, pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_smiles: dict[str, dict[str, Any]] = {}
    for row in pool_rows:
        if not row.get("valid_motif") or not row.get("canonical_motif_smiles"):
            continue
        smiles = str(row["canonical_motif_smiles"])
        current = by_smiles.get(smiles)
        if current is None:
            by_smiles[smiles] = {
                "method": method,
                "action_source": "mcs_deleted_motif",
                "motif_rank": len(by_smiles) + 1,
                "motif_smiles": row.get("deleted_motif_smiles"),
                "canonical_motif_smiles": smiles,
                "atom_count": row.get("motif_atom_count"),
                "bond_count": row.get("motif_bond_count"),
                "valid_motif": True,
                "source": "fullgraph_mcs_difference",
                "source_input_count": 1,
                "source_input_indices": str(row.get("input_idx")),
                "failure_reason": "ok",
            }
        else:
            current["source_input_count"] = int(current["source_input_count"]) + 1
            current["source_input_indices"] = f"{current['source_input_indices']}|{row.get('input_idx')}"
    rows = list(by_smiles.values())
    rows.sort(key=lambda item: (-int(item["source_input_count"]), str(item["canonical_motif_smiles"])))
    for rank, row in enumerate(rows, start=1):
        row["motif_rank"] = rank
    return rows


def morgan_fp(smiles: str) -> Any | None:
    fp, _reason = morgan_fp_with_reason(smiles)
    return fp


def morgan_fp_with_reason(smiles: str) -> tuple[Any | None, str]:
    if Chem is None or rdFingerprintGenerator is None:
        return None, "rdkit_or_fingerprint_generator_unavailable"
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return None, f"mol_parse_exception:{exc}"
    if mol is None:
        return None, "mol_parse_failed"
    try:
        generator = get_morgan_fp_generator()
        if generator is None:
            return None, "morgan_generator_unavailable"
        return generator.GetFingerprint(mol), "ok"
    except Exception as exc:
        return None, f"fingerprint_failed:{exc}"


@lru_cache(maxsize=1)
def get_morgan_fp_generator() -> Any | None:
    if rdFingerprintGenerator is None:
        return None
    try:
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    except Exception:
        return None


def tanimoto_similarity(left_smiles: str, right_smiles: str) -> float:
    if DataStructs is None:
        return 0.0
    left_fp = morgan_fp(left_smiles)
    right_fp = morgan_fp(right_smiles)
    if left_fp is None or right_fp is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
    except Exception:
        return 0.0


def pairwise_tanimoto_stats(motifs: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    smiles = [str(row["canonical_motif_smiles"]) for row in motifs if row.get("canonical_motif_smiles")]
    if len(smiles) < 2:
        return None, None
    values: list[float] = []
    for i, left in enumerate(smiles):
        for right in smiles[i + 1 :]:
            values.append(tanimoto_similarity(left, right))
    return _safe_mean(values), max(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _valid_selected_motif_smiles(motifs: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for row in motifs:
        if not row.get("valid_motif") or not row.get("canonical_motif_smiles"):
            continue
        smiles = str(row["canonical_motif_smiles"])
        if smiles in seen:
            continue
        seen.add(smiles)
        result.append(smiles)
    return result


def _motif_atom_count(smiles: str) -> int | None:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(mol.GetNumAtoms())


def _motif_has_aromatic_atom(smiles: str) -> bool:
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def _motif_has_hetero_atom(smiles: str) -> bool:
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms())


def build_motif_overlap_diagnostics(
    method_to_selected_motifs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    required = ("ours_selected_subgraph", "gt_fullgraph_greedy")
    missing = [method for method in required if method not in method_to_selected_motifs]
    if missing:
        return {
            "skipped": True,
            "reason": f"missing_methods:{','.join(missing)}",
        }

    ours = _valid_selected_motif_smiles(method_to_selected_motifs["ours_selected_subgraph"])
    gt = _valid_selected_motif_smiles(method_to_selected_motifs["gt_fullgraph_greedy"])
    if not ours or not gt:
        return {
            "skipped": True,
            "reason": "empty_valid_selected_motifs",
            "valid_motif_count_by_method": {
                "ours_selected_subgraph": len(ours),
                "gt_fullgraph_greedy": len(gt),
            },
        }

    ours_set = set(ours)
    gt_set = set(gt)
    overlap = sorted(ours_set & gt_set)
    union = ours_set | gt_set
    max_sims: list[float] = []
    fingerprint_failures: dict[str, int] = {}
    for ours_smiles in ours:
        ours_fp, ours_reason = morgan_fp_with_reason(ours_smiles)
        if ours_fp is None:
            fingerprint_failures[ours_reason] = fingerprint_failures.get(ours_reason, 0) + 1
            max_sims.append(0.0)
            continue
        best = 0.0
        for gt_smiles in gt:
            gt_fp, gt_reason = morgan_fp_with_reason(gt_smiles)
            if gt_fp is None:
                fingerprint_failures[gt_reason] = fingerprint_failures.get(gt_reason, 0) + 1
                continue
            if DataStructs is None:
                sim = 0.0
            else:
                sim = float(DataStructs.TanimotoSimilarity(ours_fp, gt_fp))
            best = max(best, sim)
        max_sims.append(best)

    atom_counts_by_method: dict[str, list[float]] = {}
    aromatic_counts: dict[str, int] = {}
    hetero_counts: dict[str, int] = {}
    for method, smiles_list in {
        "ours_selected_subgraph": ours,
        "gt_fullgraph_greedy": gt,
    }.items():
        atom_counts = [
            float(count)
            for count in (_motif_atom_count(smiles) for smiles in smiles_list)
            if count is not None
        ]
        atom_counts_by_method[method] = atom_counts
        aromatic_counts[method] = sum(1 for smiles in smiles_list if _motif_has_aromatic_atom(smiles))
        hetero_counts[method] = sum(1 for smiles in smiles_list if _motif_has_hetero_atom(smiles))

    return {
        "skipped": False,
        "exact_overlap_count": len(overlap),
        "exact_jaccard": float(len(overlap)) / float(len(union)) if union else 0.0,
        "exact_overlap_motifs": overlap,
        "mean_max_tanimoto_ours_to_gt": _safe_mean(max_sims),
        "count_ours_to_gt_sim_ge_0.7": sum(1 for value in max_sims if value >= 0.7),
        "count_ours_to_gt_sim_ge_0.85": sum(1 for value in max_sims if value >= 0.85),
        "mean_atom_count_by_method": {
            method: _safe_mean(values) for method, values in atom_counts_by_method.items()
        },
        "median_atom_count_by_method": {
            method: _safe_median(values) for method, values in atom_counts_by_method.items()
        },
        "aromatic_motif_count_by_method": aromatic_counts,
        "hetero_atom_motif_count_by_method": hetero_counts,
        "fingerprint_failure_reasons": fingerprint_failures,
    }


def evaluate_single_motif_on_record(
    *,
    record: MoleculeRecord,
    motif_row: dict[str, Any],
    target_label: int,
    teacher: RFTeacher,
) -> dict[str, Any]:
    canonical_motif = motif_row.get("canonical_motif_smiles")
    if not motif_row.get("valid_motif") or not canonical_motif:
        return {
            "match_ok": False,
            "delete_ok": False,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": motif_row.get("failure_reason") or "invalid_motif",
        }
    if not record.teacher_ok or record.p_target is None:
        return {
            "match_ok": False,
            "delete_ok": False,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": f"before_teacher_failed:{record.teacher_reason}",
        }
    try:
        match_ok = is_parent_substructure(record.canonical_smiles, str(canonical_motif))
    except Exception as exc:
        return {
            "match_ok": False,
            "delete_ok": False,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": f"substructure_check_failed:{exc}",
        }
    if not match_ok:
        return {
            "match_ok": False,
            "delete_ok": False,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": "motif_not_substructure",
        }
    deletion = delete_fragment_from_parent(record.canonical_smiles, str(canonical_motif))
    if not deletion.success or deletion.residual_smiles is None:
        return {
            "match_ok": True,
            "delete_ok": False,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": deletion.failure_reason or "motif_deletion_failed",
        }
    residual_smiles = prepare_smiles_for_oracle(deletion.residual_smiles)
    if residual_smiles is None or residual_smiles == "":
        return {
            "match_ok": True,
            "delete_ok": True,
            "valid_after": False,
            "recourse_smiles": None,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": None,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": "residual_prepare_failed",
        }
    score_after = teacher.score(residual_smiles, target_label=target_label)
    if not score_after.ok or score_after.p_target is None:
        return {
            "match_ok": True,
            "delete_ok": True,
            "valid_after": False,
            "recourse_smiles": residual_smiles,
            "p_before": record.p_target,
            "p_after": None,
            "before_teacher_label": record.teacher_label,
            "after_teacher_label": score_after.teacher_label,
            "cf_drop": None,
            "cf_flip": False,
            "motif_atom_ratio": None,
            "failure_reason": f"after_teacher_failed:{score_after.reason}",
        }
    atom_count = _float_or_none(motif_row.get("atom_count") or motif_row.get("motif_atom_count"))
    atom_ratio = float(atom_count) / float(record.atom_count) if atom_count is not None and record.atom_count else None
    cf_drop = float(record.p_target) - float(score_after.p_target)
    return {
        "match_ok": True,
        "delete_ok": True,
        "valid_after": True,
        "recourse_smiles": residual_smiles,
        "p_before": record.p_target,
        "p_after": float(score_after.p_target),
        "before_teacher_label": record.teacher_label,
        "after_teacher_label": score_after.teacher_label,
        "cf_drop": cf_drop,
        "cf_flip": teacher_strict_flip(
            record.teacher_label,
            score_after.teacher_label,
            target_label,
        ),
        "motif_atom_ratio": atom_ratio,
        "failure_reason": "ok",
    }


def evaluate_action_motifs(
    *,
    method: str,
    motif_rows: list[dict[str, Any]],
    records: list[MoleculeRecord],
    target_label: int,
    teacher: RFTeacher,
    disable_tqdm: bool,
) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    iterator = progress_iter(
        motif_rows,
        desc=f"CAMC evaluate motifs {method}",
        total=len(motif_rows),
        disable_tqdm=disable_tqdm,
    )
    for motif_row in iterator:
        motif_key = str(motif_row.get("canonical_motif_smiles") or motif_row.get("motif_smiles") or "")
        for input_idx, record in enumerate(records):
            cache_key = (record.canonical_smiles, motif_key)
            cached = cache.get(cache_key)
            if cached is None:
                cached = evaluate_single_motif_on_record(
                    record=record,
                    motif_row=motif_row,
                    target_label=target_label,
                    teacher=teacher,
                )
                cache[cache_key] = cached
            row = {
                "method": method,
                "input_idx": input_idx,
                "input_row_index": record.row_index,
                "input_id": record.molecule_id,
                "parent_smiles": record.canonical_smiles,
                "parent_label": record.label,
                "motif_rank": motif_row.get("motif_rank"),
                "motif_smiles": motif_row.get("motif_smiles"),
                "canonical_motif_smiles": motif_row.get("canonical_motif_smiles"),
            }
            row.update(cached)
            eval_rows.append(row)
    return eval_rows


def _motif_eval_maps(eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_motif: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        motif = row.get("canonical_motif_smiles")
        if not motif:
            continue
        entry = by_motif.setdefault(
            str(motif),
            {
                "support_inputs": set(),
                "flip_inputs": set(),
                "valid_cf_drops": [],
                "flip_cf_drops": [],
                "atom_ratios": [],
            },
        )
        if row.get("match_ok"):
            entry["support_inputs"].add(int(row["input_idx"]))
            if row.get("motif_atom_ratio") is not None:
                entry["atom_ratios"].append(float(row["motif_atom_ratio"]))
        if row.get("valid_after") and row.get("cf_drop") is not None:
            entry["valid_cf_drops"].append(float(row["cf_drop"]))
        if row.get("valid_after") and row.get("cf_flip") and row.get("cf_drop") is not None:
            entry["flip_inputs"].add(int(row["input_idx"]))
            entry["flip_cf_drops"].append(float(row["cf_drop"]))
    return by_motif


def select_motifs_mmr(
    *,
    method: str,
    motif_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    records: list[MoleculeRecord],
    top_k_max: int,
    reselect_ours: bool,
    disable_tqdm: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_rows = [row for row in motif_rows if row.get("valid_motif") and row.get("canonical_motif_smiles")]
    if method == "ours_selected_subgraph" and not reselect_ours:
        selected = [dict(row) for row in motif_rows[:top_k_max]]
        for rank, row in enumerate(selected, start=1):
            row["selection_rank"] = rank
            row["selection_score"] = None
            row["selection_mode"] = "original_order"
        return selected, {"selection_mode": "original_order", "candidate_count": len(motif_rows)}

    by_motif = _motif_eval_maps(eval_rows)
    candidates = {str(row["canonical_motif_smiles"]): dict(row) for row in valid_rows}
    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    mean_parent_atoms = _safe_mean([float(record.atom_count) for record in records]) or 1.0
    alpha_cf = 0.8
    beta_coverage = 20.0
    gamma_redundancy = 0.7
    eta_size = 0.3

    rounds = range(min(top_k_max, len(candidates)))
    iterator = progress_iter(rounds, desc=f"CAMC MMR select {method}", total=len(list(rounds)), disable_tqdm=disable_tqdm)
    for _ in iterator:
        best_smiles: str | None = None
        best_score: float | None = None
        best_components: dict[str, float] | None = None
        for smiles, row in candidates.items():
            stats = by_motif.get(smiles, {})
            flip_inputs = set(stats.get("flip_inputs", set()))
            coverage_gain = len(flip_inputs - covered) / float(len(records)) if records else 0.0
            valid_cf = list(stats.get("valid_cf_drops", []))
            cf_value = _safe_mean([float(value) for value in valid_cf]) or 0.0
            if selected:
                redundancy = max(
                    tanimoto_similarity(smiles, str(selected_row.get("canonical_motif_smiles")))
                    for selected_row in selected
                    if selected_row.get("canonical_motif_smiles")
                )
            else:
                redundancy = 0.0
            atom_count = _float_or_none(row.get("atom_count") or row.get("motif_atom_count")) or 0.0
            size_penalty = float(atom_count) / float(mean_parent_atoms)
            score = (
                alpha_cf * cf_value
                + beta_coverage * coverage_gain
                - gamma_redundancy * redundancy
                - eta_size * size_penalty
            )
            if best_score is None or score > best_score:
                best_score = score
                best_smiles = smiles
                best_components = {
                    "cf_value": cf_value,
                    "coverage_gain": coverage_gain,
                    "redundancy": redundancy,
                    "size_penalty": size_penalty,
                }
        if best_smiles is None or best_score is None:
            break
        selected_row = candidates.pop(best_smiles)
        selected_row["selection_rank"] = len(selected) + 1
        selected_row["selection_score"] = float(best_score)
        selected_row["selection_mode"] = "camc_mmr"
        if best_components:
            selected_row.update({f"selection_{key}": value for key, value in best_components.items()})
        selected.append(selected_row)
        covered.update(set(by_motif.get(best_smiles, {}).get("flip_inputs", set())))

    return selected, {
        "selection_mode": "camc_mmr",
        "candidate_count": len(valid_rows),
        "selected_count": len(selected),
        "alpha_cf": alpha_cf,
        "beta_coverage": beta_coverage,
        "gamma_redundancy": gamma_redundancy,
        "eta_size": eta_size,
    }


def build_camc_table_and_per_input(
    *,
    method_to_selected_motifs: dict[str, list[dict[str, Any]]],
    method_to_eval_rows: dict[str, list[dict[str, Any]]],
    records: list[MoleculeRecord],
    target_label: int,
    camc_top_k_list: list[int],
    delta_list: list[float],
    runtime_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    per_input_rows: list[dict[str, Any]] = []
    num_inputs = len(records)
    for method, selected_motifs in method_to_selected_motifs.items():
        eval_rows = method_to_eval_rows.get(method, [])
        by_input_motif: dict[tuple[int, str], dict[str, Any]] = {}
        for row in eval_rows:
            motif = row.get("canonical_motif_smiles")
            if motif:
                by_input_motif[(int(row["input_idx"]), str(motif))] = row
        for k in camc_top_k_list:
            active_motifs = selected_motifs[: min(k, len(selected_motifs))]
            active_valid = [
                row
                for row in active_motifs
                if row.get("valid_motif") and row.get("canonical_motif_smiles")
            ]
            active_smiles = [str(row["canonical_motif_smiles"]) for row in active_valid]
            support_count = 0
            flip_count = 0
            delta_counts = {float(delta): 0 for delta in delta_list}
            cf_drop_all_matched: list[float] = []
            cf_drop_covered: list[float] = []
            atom_ratios: list[float] = []
            for input_idx, record in enumerate(records):
                rows = [
                    by_input_motif[(input_idx, smiles)]
                    for smiles in active_smiles
                    if (input_idx, smiles) in by_input_motif
                ]
                support = any(row.get("match_ok") for row in rows)
                valid_rows = [row for row in rows if row.get("valid_after") and row.get("cf_drop") is not None]
                flip_rows = [row for row in valid_rows if row.get("cf_flip")]
                best_cf_all = max((float(row["cf_drop"]) for row in valid_rows), default=None)
                best_cf_flip = max((float(row["cf_drop"]) for row in flip_rows), default=None)
                if support:
                    support_count += 1
                if flip_rows:
                    flip_count += 1
                    if best_cf_flip is not None:
                        cf_drop_covered.append(best_cf_flip)
                if best_cf_all is not None:
                    cf_drop_all_matched.append(best_cf_all)
                for row in rows:
                    if row.get("motif_atom_ratio") is not None and row.get("match_ok"):
                        atom_ratios.append(float(row["motif_atom_ratio"]))
                delta_flags: dict[str, bool] = {}
                for delta in delta_list:
                    covered_delta = any(float(row["cf_drop"]) > float(delta) for row in valid_rows)
                    delta_flags[f"camc_delta_{delta}"] = covered_delta
                    if covered_delta:
                        delta_counts[float(delta)] += 1
                per_row = {
                    "method": method,
                    "target_label": int(target_label),
                    "k": int(k),
                    "input_idx": int(input_idx),
                    "input_row_index": record.row_index,
                    "input_id": record.molecule_id,
                    "parent_smiles": record.canonical_smiles,
                    "support": bool(support),
                    "camc_flip": bool(flip_rows),
                    "best_cf_drop_all_matched": best_cf_all,
                    "best_cf_drop_covered": best_cf_flip,
                }
                per_row.update(delta_flags)
                per_input_rows.append(per_row)

            tanimoto_mean, tanimoto_max = pairwise_tanimoto_stats(active_valid)
            motif_atom_counts = [
                float(row["atom_count"])
                for row in active_valid
                if row.get("atom_count") not in (None, "")
            ]
            result = {
                "method": method,
                "action_source": str(active_motifs[0].get("action_source") if active_motifs else ""),
                "target_label": int(target_label),
                "k": int(k),
                "num_inputs": int(num_inputs),
                "support_coverage": float(support_count) / float(num_inputs) if num_inputs else 0.0,
                "camc_flip_coverage": float(flip_count) / float(num_inputs) if num_inputs else 0.0,
                "mean_cf_drop_all_matched": _safe_mean(cf_drop_all_matched),
                "mean_cf_drop_covered": _safe_mean(cf_drop_covered),
                "motif_count": int(len(active_motifs)),
                "valid_motif_count": int(len(active_valid)),
                "motif_atom_count_mean": _safe_mean(motif_atom_counts),
                "motif_atom_ratio_mean": _safe_mean(atom_ratios),
                "pairwise_tanimoto_mean": tanimoto_mean,
                "pairwise_tanimoto_max": tanimoto_max,
                "runtime_sec": float(runtime_sec),
            }
            for delta, count in delta_counts.items():
                result[f"camc_delta_{delta}"] = float(count) / float(num_inputs) if num_inputs else 0.0
            table.append(result)
    return table, per_input_rows


def check_camc_monotonicity(table: list[dict[str, Any]], delta_list: list[float]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    eps = 1e-12
    metrics = ["support_coverage", "camc_flip_coverage"] + [f"camc_delta_{delta}" for delta in delta_list]
    for method in sorted({str(row["method"]) for row in table}):
        method_rows = sorted([row for row in table if str(row["method"]) == method], key=lambda row: int(row["k"]))
        for metric in metrics:
            prev_row: dict[str, Any] | None = None
            for row in method_rows:
                if metric not in row:
                    continue
                if prev_row is not None and float(row[metric]) + eps < float(prev_row[metric]):
                    warnings.append(
                        {
                            "method": method,
                            "metric": metric,
                            "k_prev": int(prev_row["k"]),
                            "k_next": int(row["k"]),
                            "value_prev": float(prev_row[metric]),
                            "value_next": float(row[metric]),
                            "warning_type": "camc_metric_decreased_with_k",
                        }
                    )
                prev_row = row
    return warnings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_lists(args: argparse.Namespace) -> tuple[list[int], list[float], list[int], list[float], list[float]]:
    top_k_list = sorted({int(value) for value in args.top_k_list if int(value) > 0})
    theta_list = sorted({float(value) for value in args.theta_list if float(value) >= 0.0})
    camc_top_k_source = args.camc_top_k_list if args.camc_top_k_list is not None else top_k_list
    camc_top_k_list = sorted({int(value) for value in camc_top_k_source if int(value) > 0})
    camc_delta_list = sorted({float(value) for value in args.camc_delta_list if float(value) >= 0.0})
    extraction_source = args.camc_extraction_theta_list if args.camc_extraction_theta_list is not None else theta_list
    camc_extraction_theta_list = sorted({float(value) for value in extraction_source if float(value) >= 0.0})
    if not top_k_list:
        raise ValueError("--top-k-list must contain at least one positive integer.")
    if not theta_list:
        raise ValueError("--theta-list must contain at least one non-negative value.")
    if not camc_top_k_list:
        raise ValueError("--camc-top-k-list must contain at least one positive integer.")
    if not camc_delta_list:
        raise ValueError("--camc-delta-list must contain at least one non-negative value.")
    return top_k_list, theta_list, camc_top_k_list, camc_delta_list, camc_extraction_theta_list


def run_camc(
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
    runtime_by_stage: dict[str, float],
    out_dir: Path,
    records: list[MoleculeRecord],
    fragments: list[FragmentRecord],
    gt_selected_fullgraphs: list[FullgraphRecord],
    teacher: RFTeacher,
    distance_proxy: DistanceProxy,
    camc_top_k_list: list[int],
    camc_delta_list: list[float],
    camc_extraction_theta_list: list[float],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    camc_start = time.time()
    method_to_motif_rows: dict[str, list[dict[str, Any]]] = {}
    method_to_pool_rows: dict[str, list[dict[str, Any]]] = {}
    method_to_selected_motifs: dict[str, list[dict[str, Any]]] = {}
    method_to_eval_rows: dict[str, list[dict[str, Any]]] = {}
    motif_extraction_stats: dict[str, Any] = {}
    motif_selection_stats: dict[str, Any] = {}

    with timed_stage("CAMC action motif extraction", logger, runtime_by_stage):
        ours_motifs = build_ours_action_motifs(
            fragments,
            min_atoms=int(args.camc_min_motif_atoms),
        )
        method_to_motif_rows["ours_selected_subgraph"] = ours_motifs
        method_to_pool_rows["ours_selected_subgraph"] = ours_motifs
        write_csv(out_dir / "camc_ours_action_motifs.csv", ours_motifs, CAMC_OURS_MOTIF_FIELDS)

        gt_pool = extract_fullgraph_action_motif_pool(
            method="gt_fullgraph_greedy",
            records=records,
            selected_fullgraphs=gt_selected_fullgraphs[: max(camc_top_k_list)],
            target_label=int(args.target_label),
            min_atoms=int(args.camc_min_motif_atoms),
            use_strict_theta=bool(args.camc_use_strict_theta),
            extraction_theta_list=camc_extraction_theta_list,
            distance_proxy=distance_proxy,
            disable_tqdm=bool(args.disable_tqdm),
        )
        gt_motifs = dedupe_fullgraph_motif_pool("gt_fullgraph_greedy", gt_pool)
        method_to_pool_rows["gt_fullgraph_greedy"] = gt_pool
        method_to_motif_rows["gt_fullgraph_greedy"] = gt_motifs
        write_csv(out_dir / "camc_gt_fullgraph_motif_pool.csv", gt_pool, CAMC_FULLGRAPH_POOL_FIELDS)

        extra_fullgraph_methods: dict[str, list[FullgraphRecord]] = {}
        for spec in args.extra_fullgraph_selected_csv:
            fullgraphs = load_extra_fullgraph_selected_csv(
                spec,
                teacher=teacher,
                target_label=int(args.target_label),
                disable_tqdm=bool(args.disable_tqdm),
            )
            if not fullgraphs:
                continue
            method = fullgraphs[0].method
            extra_fullgraph_methods[method] = fullgraphs

        for method, fullgraphs in extra_fullgraph_methods.items():
            pool = extract_fullgraph_action_motif_pool(
                method=method,
                records=records,
                selected_fullgraphs=fullgraphs[: max(camc_top_k_list)],
                target_label=int(args.target_label),
                min_atoms=int(args.camc_min_motif_atoms),
                use_strict_theta=bool(args.camc_use_strict_theta),
                extraction_theta_list=camc_extraction_theta_list,
                distance_proxy=distance_proxy,
                disable_tqdm=bool(args.disable_tqdm),
            )
            motifs = dedupe_fullgraph_motif_pool(method, pool)
            method_to_pool_rows[method] = pool
            method_to_motif_rows[method] = motifs
            safe = safe_method_name(method)
            write_csv(out_dir / f"camc_{safe}_motif_pool.csv", pool, CAMC_FULLGRAPH_POOL_FIELDS)

        for method, pool_rows in method_to_pool_rows.items():
            motif_extraction_stats[method] = {
                "pool_row_count": len(pool_rows),
                "valid_pool_row_count": sum(1 for row in pool_rows if row.get("valid_motif")),
                "unique_valid_motif_count": len(method_to_motif_rows.get(method, [])),
            }

    with timed_stage("CAMC action motif support / flip / cf_drop precompute", logger, runtime_by_stage):
        for method, motif_rows in method_to_motif_rows.items():
            eval_rows = evaluate_action_motifs(
                method=method,
                motif_rows=motif_rows,
                records=records,
                target_label=int(args.target_label),
                teacher=teacher,
                disable_tqdm=bool(args.disable_tqdm),
            )
            method_to_eval_rows[method] = eval_rows

    with timed_stage("CAMC action motif selection", logger, runtime_by_stage):
        for method, motif_rows in method_to_motif_rows.items():
            selected, stats = select_motifs_mmr(
                method=method,
                motif_rows=motif_rows,
                eval_rows=method_to_eval_rows.get(method, []),
                records=records,
                top_k_max=max(camc_top_k_list),
                reselect_ours=bool(args.camc_reselect_ours_motifs),
                disable_tqdm=bool(args.disable_tqdm),
            )
            method_to_selected_motifs[method] = selected
            motif_selection_stats[method] = stats
            if method == "ours_selected_subgraph":
                continue
            safe = safe_method_name(method)
            write_csv(out_dir / f"camc_{safe}_selected_motifs.csv", selected, CAMC_SELECTED_MOTIF_FIELDS)
        write_csv(
            out_dir / "camc_gt_fullgraph_selected_motifs.csv",
            method_to_selected_motifs.get("gt_fullgraph_greedy", []),
            CAMC_SELECTED_MOTIF_FIELDS,
        )
        motif_overlap_diagnostics = build_motif_overlap_diagnostics(method_to_selected_motifs)

    with timed_stage("CAMC action motif evaluation", logger, runtime_by_stage):
        camc_runtime = time.time() - camc_start
        camc_table, camc_per_input = build_camc_table_and_per_input(
            method_to_selected_motifs=method_to_selected_motifs,
            method_to_eval_rows=method_to_eval_rows,
            records=records,
            target_label=int(args.target_label),
            camc_top_k_list=camc_top_k_list,
            delta_list=camc_delta_list,
            runtime_sec=camc_runtime,
        )
        camc_warnings = check_camc_monotonicity(camc_table, camc_delta_list)
        write_csv(out_dir / "camc_comparison_table.csv", camc_table)
        write_csv(out_dir / "camc_per_input.csv", camc_per_input)

    output_paths = {
        "camc_comparison_table_csv": str(out_dir / "camc_comparison_table.csv"),
        "camc_summary_json": str(out_dir / "camc_summary.json"),
        "camc_per_input_csv": str(out_dir / "camc_per_input.csv"),
        "camc_ours_action_motifs_csv": str(out_dir / "camc_ours_action_motifs.csv"),
        "camc_gt_fullgraph_motif_pool_csv": str(out_dir / "camc_gt_fullgraph_motif_pool.csv"),
        "camc_gt_fullgraph_selected_motifs_csv": str(out_dir / "camc_gt_fullgraph_selected_motifs.csv"),
    }
    for method in method_to_pool_rows:
        if method in ("ours_selected_subgraph", "gt_fullgraph_greedy"):
            continue
        safe = safe_method_name(method)
        output_paths[f"camc_{safe}_motif_pool_csv"] = str(out_dir / f"camc_{safe}_motif_pool.csv")
        output_paths[f"camc_{safe}_selected_motifs_csv"] = str(out_dir / f"camc_{safe}_selected_motifs.csv")

    camc_summary = {
        "run_config": {
            "target_label": int(args.target_label),
            "camc_delta_list": camc_delta_list,
            "camc_top_k_list": camc_top_k_list,
            "camc_min_motif_atoms": int(args.camc_min_motif_atoms),
            "camc_use_strict_theta": bool(args.camc_use_strict_theta),
            "camc_extraction_theta_list": camc_extraction_theta_list,
            "camc_reselect_ours_motifs": bool(args.camc_reselect_ours_motifs),
            "extra_fullgraph_selected_csv": list(args.extra_fullgraph_selected_csv),
        },
        "method_summaries": {
            method: [row for row in camc_table if row["method"] == method]
            for method in sorted({row["method"] for row in camc_table})
        },
        "motif_extraction_stats": motif_extraction_stats,
        "motif_selection_stats": motif_selection_stats,
        "motif_overlap_diagnostics": motif_overlap_diagnostics,
        "camc_monotonicity_warnings": camc_warnings,
        "output_paths": output_paths,
    }
    write_json(out_dir / "camc_summary.json", camc_summary)

    motif_counts = {
        method: len(rows)
        for method, rows in method_to_motif_rows.items()
    }
    invalid_counts = {
        method: sum(1 for row in method_to_pool_rows.get(method, []) if not row.get("valid_motif"))
        for method in method_to_pool_rows
    }
    return camc_summary, motif_counts, invalid_counts, camc_warnings, motif_overlap_diagnostics


def main() -> int:
    args = parse_args()
    start_time = time.time()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)
    configure_rdkit_warnings(suppress=bool(args.suppress_rdkit_warnings), logger=logger)
    runtime_by_stage: dict[str, float] = {}

    if Chem is None or rdFMCS is None:
        raise RuntimeError("RDKit is required for HIV quick recourse comparison.")

    with timed_stage("load config", logger, runtime_by_stage):
        top_k_list, theta_list, camc_top_k_list, camc_delta_list, camc_extraction_theta_list = normalize_lists(args)
        progress_every = max(1, int(args.progress_every))
        max_recourse_k = max(top_k_list)
        max_fullgraph_selection_k = max(max(top_k_list), max(camc_top_k_list))
        logger.info(
            "config target_label=%s top_k_list=%s theta_list=%s enable_camc=%s camc_top_k_list=%s",
            args.target_label,
            top_k_list,
            theta_list,
            args.enable_camc,
            camc_top_k_list,
        )

    with timed_stage("load HIV CSV", logger, runtime_by_stage):
        raw_rows, resolved_smiles_col, resolved_label_col = load_hiv_rows(
            args.hiv_csv,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
        )
        logger.info(
            "loaded_hiv_csv path=%s rows=%d smiles_col=%s label_col=%s",
            args.hiv_csv,
            len(raw_rows),
            resolved_smiles_col,
            resolved_label_col,
        )

    with timed_stage("parse SMILES", logger, runtime_by_stage):
        valid_records, parse_metadata = parse_hiv_records(
            raw_rows,
            resolved_smiles_col=resolved_smiles_col,
            resolved_label_col=resolved_label_col,
            disable_tqdm=bool(args.disable_tqdm),
        )
        logger.info("parsed_smiles valid_records=%d metadata=%s", len(valid_records), parse_metadata)

    with timed_stage("load teacher", logger, runtime_by_stage):
        teacher = RFTeacher(args.teacher_path)
        distance_proxy = DistanceProxy(timeout_sec=int(args.mcs_timeout_sec))
        logger.info("loaded_teacher path=%s classes=%s", teacher.teacher_path, teacher.classes)

    with timed_stage("load selected fragments", logger, runtime_by_stage):
        fragments, selected_files = load_selected_fragments(args.ours_selected_dir)
        if len(fragments) < max_recourse_k:
            logger.warning(
                "only %d selected fragments parsed, less than requested max k=%d",
                len(fragments),
                max_recourse_k,
            )
        logger.info("selected_fragment_count=%d", len(fragments))

    with timed_stage("prepare target inputs", logger, runtime_by_stage):
        target_records = prepare_target_inputs(
            valid_records,
            target_label=int(args.target_label),
            teacher=teacher,
            max_inputs=args.max_inputs,
            seed=int(args.seed),
            disable_tqdm=bool(args.disable_tqdm),
        )
        if not target_records:
            raise ValueError(f"No valid target-label inputs found for target_label={args.target_label}.")
        logger.info("target_input_count=%d", len(target_records))

    with timed_stage("prepare opposite-label candidates", logger, runtime_by_stage):
        gt_candidates, opposite_before_sampling = prepare_opposite_candidates(
            valid_records,
            target_label=int(args.target_label),
            teacher=teacher,
            max_gt_candidates=int(args.max_gt_candidates),
            seed=int(args.seed),
            disable_tqdm=bool(args.disable_tqdm),
        )
        if not gt_candidates:
            raise ValueError(f"No valid opposite-label GT candidates found for target_label={args.target_label}.")
        logger.info(
            "gt_candidate_count=%d opposite_before_sampling=%d",
            len(gt_candidates),
            opposite_before_sampling,
        )

    with timed_stage("evaluate ours action candidates", logger, runtime_by_stage):
        ours_action_rows, ours_progress_stats = evaluate_ours_action_candidates(
            target_records,
            fragments,
            top_k_max=max_recourse_k,
            target_label=int(args.target_label),
            teacher=teacher,
            distance_proxy=distance_proxy,
            progress_every=progress_every,
            disable_tqdm=bool(args.disable_tqdm),
            logger=logger,
        )
        ours_rows = aggregate_ours_recourse_rows(
            target_records,
            ours_action_rows,
            top_k_list=top_k_list,
            theta_list=theta_list,
            target_label=int(args.target_label),
        )
        (
            per_k_match_rate,
            per_k_delete_ok_rate,
            per_k_any_flip_rate,
            per_k_any_feasible_by_theta,
        ) = build_ours_diagnostics(
            ours_action_rows,
            records=target_records,
            top_k_list=top_k_list,
            theta_list=theta_list,
        )

    with timed_stage("evaluate GT fullgraph candidates", logger, runtime_by_stage):
        gt_candidate_matrix, eligible_gt_indices = build_gt_candidate_matrix(
            target_records,
            gt_candidates,
            target_label=int(args.target_label),
            distance_proxy=distance_proxy,
            disable_tqdm=bool(args.disable_tqdm),
        )
        logger.info(
            "eligible_gt_candidates=%d pairwise_ok_entries=%d",
            len(eligible_gt_indices),
            len(gt_candidate_matrix),
        )

    with timed_stage("greedy select GT fullgraphs", logger, runtime_by_stage):
        gt_selected_indices, gt_selected_rows = greedy_select_gt_fullgraphs(
            target_records,
            gt_candidates,
            candidate_matrix=gt_candidate_matrix,
            eligible_candidate_indices=eligible_gt_indices,
            theta_list=theta_list,
            target_label=int(args.target_label),
            top_k_max=max_fullgraph_selection_k,
            disable_tqdm=bool(args.disable_tqdm),
        )
        gt_rows = aggregate_gt_recourse_rows(
            target_records,
            gt_candidates,
            selected_indices=gt_selected_indices,
            candidate_matrix=gt_candidate_matrix,
            top_k_list=top_k_list,
            theta_list=theta_list,
            target_label=int(args.target_label),
        )
        gt_selected_fullgraphs = [
            molecule_record_to_fullgraph(
                gt_candidates[candidate_index],
                method="gt_fullgraph_greedy",
                rank=rank,
                source_index=candidate_index,
                source_path=str(args.hiv_csv),
            )
            for rank, candidate_index in enumerate(gt_selected_indices, start=1)
        ]

    with timed_stage("recourse-level aggregate metrics", logger, runtime_by_stage):
        runtime_sec = time.time() - start_time
        comparison_table = build_comparison_table(
            {
                "ours_selected_subgraph": ours_rows,
                "gt_fullgraph_greedy": gt_rows,
            },
            target_label=int(args.target_label),
            top_k_list=top_k_list,
            theta_list=theta_list,
            num_inputs=len(target_records),
            runtime_sec=runtime_sec,
        )
        recourse_warnings = check_recourse_monotonicity(comparison_table)
        logger.info("recourse_monotonicity_warnings=%d", len(recourse_warnings))

    camc_summary: dict[str, Any] | None = None
    camc_motif_counts_by_method: dict[str, int] = {}
    camc_invalid_motif_counts_by_method: dict[str, int] = {}
    camc_warnings: list[dict[str, Any]] = []
    motif_overlap_diagnostics: dict[str, Any] = {
        "skipped": True,
        "reason": "camc_disabled",
    }
    if args.enable_camc:
        (
            camc_summary,
            camc_motif_counts_by_method,
            camc_invalid_motif_counts_by_method,
            camc_warnings,
            motif_overlap_diagnostics,
        ) = run_camc(
            args=args,
            logger=logger,
            runtime_by_stage=runtime_by_stage,
            out_dir=out_dir,
            records=target_records,
            fragments=fragments,
            gt_selected_fullgraphs=gt_selected_fullgraphs,
            teacher=teacher,
            distance_proxy=distance_proxy,
            camc_top_k_list=camc_top_k_list,
            camc_delta_list=camc_delta_list,
            camc_extraction_theta_list=camc_extraction_theta_list,
        )
    else:
        logger.info("CAMC disabled by --disable-camc")

    with timed_stage("write outputs", logger, runtime_by_stage):
        data_metadata = {
            "hiv_csv": str(args.hiv_csv),
            "smiles_col": resolved_smiles_col,
            "label_col": resolved_label_col,
            "target_record_count_before_sampling": len([record for record in valid_records if record.label == int(args.target_label)]),
            "opposite_candidate_count_before_sampling": opposite_before_sampling,
            "target_record_count": len(target_records),
            "opposite_candidate_count": len(gt_candidates),
        }
        data_metadata.update(parse_metadata)
        run_config = {
            "hiv_csv": str(args.hiv_csv.expanduser().resolve()),
            "teacher_path": str(args.teacher_path.expanduser().resolve()),
            "target_label": int(args.target_label),
            "smiles_col": args.smiles_col,
            "label_col": args.label_col,
            "ours_selected_dir": str(args.ours_selected_dir.expanduser().resolve()),
            "top_k_list": top_k_list,
            "theta_list": theta_list,
            "max_inputs": args.max_inputs,
            "max_gt_candidates": int(args.max_gt_candidates),
            "out_dir": str(out_dir),
            "seed": int(args.seed),
            "config": args.config,
            "set_overrides": args.set,
            "mcs_timeout_sec": int(args.mcs_timeout_sec),
            "progress_every": progress_every,
            "disable_tqdm": bool(args.disable_tqdm),
            "suppress_rdkit_warnings": bool(args.suppress_rdkit_warnings),
            "enable_camc": bool(args.enable_camc),
            "camc_delta_list": camc_delta_list,
            "camc_top_k_list": camc_top_k_list,
            "camc_min_motif_atoms": int(args.camc_min_motif_atoms),
            "camc_use_strict_theta": bool(args.camc_use_strict_theta),
            "camc_extraction_theta_list": camc_extraction_theta_list,
            "extra_fullgraph_selected_csv": list(args.extra_fullgraph_selected_csv),
        }
        output_paths = {
            "comparison_summary_json": str(out_dir / "comparison_summary.json"),
            "comparison_table_csv": str(out_dir / "comparison_table.csv"),
            "ours_per_input_csv": str(out_dir / "ours_per_input.csv"),
            "gt_fullgraph_per_input_csv": str(out_dir / "gt_fullgraph_per_input.csv"),
            "gt_selected_fullgraphs_csv": str(out_dir / "gt_selected_fullgraphs.csv"),
            "ours_action_candidates_csv": str(out_dir / "ours_action_candidates.csv"),
            "diagnostic_counts_json": str(out_dir / "diagnostic_counts.json"),
            "progress_log": str(out_dir / "progress.log"),
            "run_config_json": str(out_dir / "run_config.json"),
        }
        if args.enable_camc:
            output_paths.update(
                {
                    "camc_comparison_table_csv": str(out_dir / "camc_comparison_table.csv"),
                    "camc_summary_json": str(out_dir / "camc_summary.json"),
                    "camc_per_input_csv": str(out_dir / "camc_per_input.csv"),
                    "camc_ours_action_motifs_csv": str(out_dir / "camc_ours_action_motifs.csv"),
                    "camc_gt_fullgraph_motif_pool_csv": str(out_dir / "camc_gt_fullgraph_motif_pool.csv"),
                    "camc_gt_fullgraph_selected_motifs_csv": str(out_dir / "camc_gt_fullgraph_selected_motifs.csv"),
                }
            )
        summary = {
            "run_config": run_config,
            "data": data_metadata,
            "selected_files": selected_files,
            "selected_fragment_count": len(fragments),
            "selected_fragments": [
                {"rank": item.rank, "fragment": item.fragment, "source": item.source}
                for item in fragments
            ],
            "ours_substructure_match_rate_by_k": per_k_match_rate,
            "comparison_table": comparison_table,
            "recourse_monotonicity_warnings": recourse_warnings,
            "outputs": output_paths,
        }
        diagnostic_counts = {
            "num_inputs": len(target_records),
            "selected_fragment_count": len(fragments),
            "gt_candidate_count": len(gt_candidates),
            "runtime_by_stage": runtime_by_stage,
            "mcs_statistics": distance_proxy.statistics(),
            "per_k_match_rate": per_k_match_rate,
            "per_k_delete_ok_rate": per_k_delete_ok_rate,
            "per_k_any_flip_rate": per_k_any_flip_rate,
            "per_k_any_feasible_by_theta": per_k_any_feasible_by_theta,
            "ours_progress_stats": ours_progress_stats,
            "recourse_monotonicity_warnings": recourse_warnings,
            "camc_monotonicity_warnings": camc_warnings,
            "camc_motif_counts_by_method": camc_motif_counts_by_method,
            "camc_invalid_motif_counts_by_method": camc_invalid_motif_counts_by_method,
            "motif_overlap_diagnostics": motif_overlap_diagnostics,
        }
        write_json(out_dir / "run_config.json", run_config)
        write_json(out_dir / "comparison_summary.json", summary)
        write_json(out_dir / "diagnostic_counts.json", diagnostic_counts)
        write_csv(out_dir / "comparison_table.csv", comparison_table, COMPARISON_FIELDS)
        write_csv(out_dir / "ours_per_input.csv", ours_rows, PER_INPUT_FIELDS)
        write_csv(out_dir / "gt_fullgraph_per_input.csv", gt_rows, PER_INPUT_FIELDS)
        write_csv(out_dir / "gt_selected_fullgraphs.csv", gt_selected_rows)
        write_csv(out_dir / "ours_action_candidates.csv", ours_action_rows, OURS_ACTION_FIELDS)

    logger.info("[HIV_COMPARE] out_dir=%s", out_dir)
    logger.info("[HIV_COMPARE] num_inputs=%d", len(target_records))
    logger.info("[HIV_COMPARE] gt_candidates=%d", len(gt_candidates))
    logger.info("[HIV_COMPARE] selected_fragments=%d", len(fragments))
    logger.info("[HIV_COMPARE] comparison_table=%s", out_dir / "comparison_table.csv")
    logger.info("[HIV_COMPARE] comparison_summary=%s", out_dir / "comparison_summary.json")
    if args.enable_camc:
        logger.info("[HIV_COMPARE] camc_comparison_table=%s", out_dir / "camc_comparison_table.csv")
        logger.info("[HIV_COMPARE] camc_summary=%s", out_dir / "camc_summary.json")
    if "diagnostic_counts" in locals():
        diagnostic_counts["runtime_by_stage"] = runtime_by_stage
        write_json(out_dir / "diagnostic_counts.json", diagnostic_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
