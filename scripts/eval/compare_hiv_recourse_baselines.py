#!/usr/bin/env python3
"""Quick recourse-level comparison for HIV/SMILES counterfactual baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chem.deletion import delete_fragment_from_parent  # noqa: E402
from src.chem.substructure import is_parent_substructure  # noqa: E402
from src.rewards.reward_calculator import (  # noqa: E402
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)

try:  # pragma: no cover - depends on runtime env
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
except ImportError:  # pragma: no cover - depends on runtime env
    Chem = None
    rdFMCS = None


SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "canonical_smiles", "mol", "molecule")
LABEL_COLUMN_CANDIDATES = ("label", "y", "HIV_active", "active")
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
    "median_cost",
    "mean_cost",
    "mean_cf_drop",
    "flip_rate",
    "valid_recourse_rate",
    "runtime_sec",
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
class RecourseChoice:
    valid: bool
    reason: str
    recourse_smiles: str | None = None
    fragment_rank: int | None = None
    fragment_smiles: str | None = None
    candidate_rank: int | None = None
    candidate_index: int | None = None
    candidate_label: int | None = None
    p_before: float | None = None
    p_after: float | None = None
    before_teacher_label: int | None = None
    after_teacher_label: int | None = None
    cf_drop: float | None = None
    cf_flip: bool = False
    cost: float | None = None
    proxy_edit: float | None = None


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
                        f"Could not load teacher bundle via project loader, joblib, or pickle: "
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
    """RDKit MCS proxy distance with a small in-memory cache."""

    def __init__(self, *, timeout_sec: int = 1) -> None:
        if Chem is None or rdFMCS is None:
            raise RuntimeError("RDKit is required for MCS proxy distance.")
        self.timeout_sec = int(timeout_sec)
        self._mol_cache: dict[str, Any] = {}
        self._distance_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def mol(self, smiles: str) -> Any | None:
        if smiles not in self._mol_cache:
            self._mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        return self._mol_cache[smiles]

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
        key = tuple(sorted((left_smiles, right_smiles)))
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        left_mol = self.mol(left_smiles)
        right_mol = self.mol(right_smiles)
        if left_mol is None or right_mol is None:
            result = {
                "ok": False,
                "cost": None,
                "proxy_edit": None,
                "mcs_atoms": None,
                "mcs_bonds": None,
                "reason": "distance_parse_failed",
            }
            self._distance_cache[key] = result
            return result

        try:
            mcs = rdFMCS.FindMCS(
                [left_mol, right_mol],
                timeout=self.timeout_sec,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
                matchValences=False,
            )
            mcs_atoms = int(mcs.numAtoms)
            mcs_bonds = int(mcs.numBonds)
        except Exception as exc:  # pragma: no cover - RDKit-specific
            result = {
                "ok": False,
                "cost": None,
                "proxy_edit": None,
                "mcs_atoms": None,
                "mcs_bonds": None,
                "reason": f"mcs_failed:{exc}",
            }
            self._distance_cache[key] = result
            return result

        atoms1 = int(left_mol.GetNumAtoms())
        atoms2 = int(right_mol.GetNumAtoms())
        bonds1 = int(left_mol.GetNumBonds())
        bonds2 = int(right_mol.GetNumBonds())
        proxy_edit = (atoms1 - mcs_atoms) + (atoms2 - mcs_atoms) + (bonds1 - mcs_bonds) + (bonds2 - mcs_bonds)
        denominator = max(1, atoms1 + atoms2 + bonds1 + bonds2)
        result = {
            "ok": True,
            "cost": float(proxy_edit) / float(denominator),
            "proxy_edit": float(proxy_edit),
            "mcs_atoms": mcs_atoms,
            "mcs_bonds": mcs_bonds,
            "reason": "ok",
        }
        self._distance_cache[key] = result
        return result


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
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
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
) -> list[MoleculeRecord]:
    scored: list[MoleculeRecord] = []
    for record in records:
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


def load_hiv_records(
    csv_path: Path,
    *,
    smiles_col: str | None,
    label_col: str | None,
    target_label: int,
    teacher: RFTeacher,
    max_inputs: int | None,
    max_gt_candidates: int,
    seed: int,
) -> tuple[list[MoleculeRecord], list[MoleculeRecord], dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"HIV CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"HIV CSV has no header: {csv_path}")
        resolved_smiles_col = _resolve_column(reader.fieldnames, smiles_col, SMILES_COLUMN_CANDIDATES, "SMILES")
        resolved_label_col = _resolve_column(reader.fieldnames, label_col, LABEL_COLUMN_CANDIDATES, "label")
        raw_rows = list(reader)

    valid_records: list[BasicMoleculeRecord] = []
    invalid_smiles_count = 0
    invalid_label_count = 0
    for row_index, row in enumerate(raw_rows):
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

    target_basic = [record for record in valid_records if record.label == int(target_label)]
    target_basic = _sample_basic_records(target_basic, max_inputs, seed)

    opposite_seen: set[str] = set()
    opposite_basic: list[BasicMoleculeRecord] = []
    for record in valid_records:
        if record.label == int(target_label):
            continue
        if record.canonical_smiles in opposite_seen:
            continue
        opposite_seen.add(record.canonical_smiles)
        opposite_basic.append(record)

    opposite_basic = _sample_basic_records(opposite_basic, max_gt_candidates, seed + 17)
    target_records = _score_basic_records(
        target_basic,
        teacher=teacher,
        target_label=target_label,
    )
    opposite_records = _score_basic_records(
        opposite_basic,
        teacher=teacher,
        target_label=target_label,
    )
    metadata = {
        "hiv_csv": str(csv_path),
        "smiles_col": resolved_smiles_col,
        "label_col": resolved_label_col,
        "raw_row_count": len(raw_rows),
        "valid_record_count": len(valid_records),
        "invalid_smiles_count": invalid_smiles_count,
        "invalid_label_count": invalid_label_count,
        "target_record_count_before_sampling": len([record for record in valid_records if record.label == int(target_label)]),
        "opposite_candidate_count_before_sampling": len(opposite_seen),
        "target_record_count": len(target_records),
        "opposite_candidate_count": len(opposite_records),
    }
    return target_records, opposite_records, metadata


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


def choose_ours_recourse(
    record: MoleculeRecord,
    fragments: list[FragmentRecord],
    *,
    target_label: int,
    teacher: RFTeacher,
    distance_proxy: DistanceProxy,
) -> RecourseChoice:
    if not record.teacher_ok or record.p_target is None:
        return RecourseChoice(
            valid=False,
            reason=f"before_teacher_failed:{record.teacher_reason}",
            p_before=record.p_target,
            before_teacher_label=record.teacher_label,
        )

    best: RecourseChoice | None = None
    saw_substructure = False
    saw_deletion = False
    for fragment in fragments:
        if not is_parent_substructure(record.canonical_smiles, fragment.fragment):
            continue
        saw_substructure = True
        deletion = delete_fragment_from_parent(record.canonical_smiles, fragment.fragment)
        if not deletion.success or deletion.residual_smiles is None:
            continue
        saw_deletion = True
        residual_smiles = prepare_smiles_for_oracle(deletion.residual_smiles)
        if residual_smiles is None or residual_smiles == "":
            continue
        score_after = teacher.score(residual_smiles, target_label=target_label)
        if not score_after.ok or score_after.p_target is None:
            continue
        distance = distance_proxy.distance(record.canonical_smiles, residual_smiles)
        if not distance["ok"] or distance["cost"] is None:
            continue
        cf_drop = float(record.p_target) - float(score_after.p_target)
        choice = RecourseChoice(
            valid=True,
            reason="ok",
            recourse_smiles=residual_smiles,
            fragment_rank=fragment.rank,
            fragment_smiles=fragment.fragment,
            p_before=float(record.p_target),
            p_after=float(score_after.p_target),
            before_teacher_label=record.teacher_label,
            after_teacher_label=score_after.teacher_label,
            cf_drop=cf_drop,
            cf_flip=(score_after.teacher_label != int(target_label)),
            cost=float(distance["cost"]),
            proxy_edit=float(distance["proxy_edit"]),
        )
        if best is None:
            best = choice
        else:
            best_key = (float(best.cf_drop or -1e9), -float(best.cost or 1e9))
            choice_key = (float(choice.cf_drop or -1e9), -float(choice.cost or 1e9))
            if choice_key > best_key:
                best = choice

    if best is not None:
        return best
    if saw_deletion:
        reason = "no_scorable_deleted_residual"
    elif saw_substructure:
        reason = "substructure_found_but_deletion_failed"
    else:
        reason = "no_selected_fragment_substructure"
    return RecourseChoice(
        valid=False,
        reason=reason,
        p_before=record.p_target,
        before_teacher_label=record.teacher_label,
    )


def recourse_choice_to_row(
    *,
    method: str,
    target_label: int,
    k: int,
    theta: float,
    record: MoleculeRecord,
    choice: RecourseChoice,
) -> dict[str, Any]:
    covered = bool(choice.valid and choice.cf_flip and choice.cost is not None and float(choice.cost) <= float(theta))
    return {
        "method": method,
        "target_label": int(target_label),
        "k": int(k),
        "theta": float(theta),
        "input_row_index": record.row_index,
        "input_id": record.molecule_id,
        "parent_smiles": record.canonical_smiles,
        "parent_label": record.label,
        "p_before": choice.p_before,
        "before_teacher_label": choice.before_teacher_label,
        "recourse_smiles": choice.recourse_smiles,
        "fragment_rank": choice.fragment_rank,
        "fragment_smiles": choice.fragment_smiles,
        "candidate_rank": choice.candidate_rank,
        "candidate_index": choice.candidate_index,
        "candidate_label": choice.candidate_label,
        "p_after": choice.p_after,
        "after_teacher_label": choice.after_teacher_label,
        "cf_drop": choice.cf_drop,
        "cf_flip": bool(choice.cf_flip),
        "cost": choice.cost,
        "proxy_edit": choice.proxy_edit,
        "valid_recourse": bool(choice.valid),
        "covered": covered,
        "reason": choice.reason,
    }


def evaluate_ours(
    records: list[MoleculeRecord],
    fragments: list[FragmentRecord],
    *,
    top_k_list: list[int],
    theta_list: list[float],
    target_label: int,
    teacher: RFTeacher,
    distance_proxy: DistanceProxy,
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    rows: list[dict[str, Any]] = []
    substructure_rates: dict[int, float] = {}
    max_k = max(top_k_list) if top_k_list else len(fragments)
    cached_by_k: dict[int, list[RecourseChoice]] = {}

    for k in sorted(set(top_k_list)):
        active_fragments = fragments[: min(k, len(fragments))]
        substructure_hits = 0
        choices: list[RecourseChoice] = []
        for record in records:
            if any(is_parent_substructure(record.canonical_smiles, fragment.fragment) for fragment in active_fragments):
                substructure_hits += 1
            choices.append(
                choose_ours_recourse(
                    record,
                    active_fragments,
                    target_label=target_label,
                    teacher=teacher,
                    distance_proxy=distance_proxy,
                )
            )
        cached_by_k[k] = choices
        substructure_rates[k] = float(substructure_hits) / float(len(records)) if records else 0.0

    del max_k
    for k, choices in cached_by_k.items():
        for theta in theta_list:
            for record, choice in zip(records, choices):
                rows.append(
                    recourse_choice_to_row(
                        method="ours_selected_subgraph",
                        target_label=target_label,
                        k=k,
                        theta=theta,
                        record=record,
                        choice=choice,
                    )
                )
    return rows, substructure_rates


def _choice_from_gt_candidate(
    record: MoleculeRecord,
    candidate: MoleculeRecord | None,
    *,
    candidate_rank: int | None,
    candidate_index: int | None,
    target_label: int,
    distance_proxy: DistanceProxy,
) -> RecourseChoice:
    if candidate is None:
        return RecourseChoice(
            valid=False,
            reason="no_selected_candidate",
            p_before=record.p_target,
            before_teacher_label=record.teacher_label,
        )
    if not record.teacher_ok or record.p_target is None:
        return RecourseChoice(
            valid=False,
            reason=f"before_teacher_failed:{record.teacher_reason}",
            p_before=record.p_target,
            before_teacher_label=record.teacher_label,
            candidate_rank=candidate_rank,
            candidate_index=candidate_index,
            candidate_label=candidate.label,
        )
    if not candidate.teacher_ok or candidate.p_target is None:
        return RecourseChoice(
            valid=False,
            reason=f"candidate_teacher_failed:{candidate.teacher_reason}",
            p_before=record.p_target,
            before_teacher_label=record.teacher_label,
            candidate_rank=candidate_rank,
            candidate_index=candidate_index,
            candidate_label=candidate.label,
            recourse_smiles=candidate.canonical_smiles,
        )
    distance = distance_proxy.distance(record.canonical_smiles, candidate.canonical_smiles)
    if not distance["ok"] or distance["cost"] is None:
        return RecourseChoice(
            valid=False,
            reason=distance["reason"],
            p_before=record.p_target,
            before_teacher_label=record.teacher_label,
            candidate_rank=candidate_rank,
            candidate_index=candidate_index,
            candidate_label=candidate.label,
            recourse_smiles=candidate.canonical_smiles,
        )
    return RecourseChoice(
        valid=True,
        reason="ok",
        recourse_smiles=candidate.canonical_smiles,
        candidate_rank=candidate_rank,
        candidate_index=candidate_index,
        candidate_label=candidate.label,
        p_before=float(record.p_target),
        p_after=float(candidate.p_target),
        before_teacher_label=record.teacher_label,
        after_teacher_label=candidate.teacher_label,
        cf_drop=float(record.p_target) - float(candidate.p_target),
        cf_flip=(candidate.teacher_label != int(target_label)),
        cost=float(distance["cost"]),
        proxy_edit=float(distance["proxy_edit"]),
    )


def build_gt_greedy_orders(
    records: list[MoleculeRecord],
    candidates: list[MoleculeRecord],
    *,
    theta_list: list[float],
    target_label: int,
    top_k_max: int,
    distance_proxy: DistanceProxy,
) -> tuple[dict[float, list[int]], list[dict[str, Any]]]:
    coverage_sets: dict[float, list[set[int]]] = {
        theta: [set() for _ in candidates] for theta in theta_list
    }
    cf_drop_sums: dict[float, list[float]] = {
        theta: [0.0 for _ in candidates] for theta in theta_list
    }
    cost_sums: dict[float, list[float]] = {
        theta: [0.0 for _ in candidates] for theta in theta_list
    }

    for candidate_index, candidate in enumerate(candidates):
        if not candidate.teacher_ok or candidate.p_target is None or candidate.teacher_label == int(target_label):
            continue
        for input_index, record in enumerate(records):
            if not record.teacher_ok or record.p_target is None:
                continue
            distance = distance_proxy.distance(record.canonical_smiles, candidate.canonical_smiles)
            if not distance["ok"] or distance["cost"] is None:
                continue
            cost = float(distance["cost"])
            cf_drop = float(record.p_target) - float(candidate.p_target)
            for theta in theta_list:
                if cost <= float(theta):
                    coverage_sets[theta][candidate_index].add(input_index)
                    cf_drop_sums[theta][candidate_index] += cf_drop
                    cost_sums[theta][candidate_index] += cost

    orders: dict[float, list[int]] = {}
    selected_rows: list[dict[str, Any]] = []
    num_inputs = len(records)

    for theta in theta_list:
        selected: list[int] = []
        covered: set[int] = set()
        remaining = {
            index
            for index, candidate in enumerate(candidates)
            if candidate.teacher_ok and candidate.teacher_label != int(target_label)
        }
        while len(selected) < top_k_max and remaining:
            best_index: int | None = None
            best_key: tuple[float, float, float, float] | None = None
            for candidate_index in remaining:
                new_covered = coverage_sets[theta][candidate_index] - covered
                gain = len(new_covered)
                if gain > 0:
                    cf_gain = sum(
                        float(records[input_index].p_target or 0.0) - float(candidates[candidate_index].p_target or 0.0)
                        for input_index in new_covered
                    )
                    cost_gain = sum(
                        float(
                            distance_proxy.distance(
                                records[input_index].canonical_smiles,
                                candidates[candidate_index].canonical_smiles,
                            )["cost"]
                            or 0.0
                        )
                        for input_index in new_covered
                    )
                    mean_cost_gain = cost_gain / gain
                else:
                    cf_gain = 0.0
                    mean_cost_gain = 1e9
                key = (float(gain), float(cf_gain), -float(mean_cost_gain), -float(candidate_index))
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = candidate_index
            if best_index is None:
                break
            selected.append(best_index)
            remaining.remove(best_index)
            new_covered = coverage_sets[theta][best_index] - covered
            covered.update(coverage_sets[theta][best_index])
            rank = len(selected)
            gain_count = len(new_covered)
            selected_rows.append(
                {
                    "method": "gt_fullgraph_greedy",
                    "target_label": int(target_label),
                    "theta": float(theta),
                    "rank": rank,
                    "candidate_index": best_index,
                    "candidate_row_index": candidates[best_index].row_index,
                    "candidate_id": candidates[best_index].molecule_id,
                    "candidate_smiles": candidates[best_index].canonical_smiles,
                    "candidate_label": candidates[best_index].label,
                    "candidate_p_target": candidates[best_index].p_target,
                    "candidate_teacher_label": candidates[best_index].teacher_label,
                    "coverage_gain_count": gain_count,
                    "coverage_gain": float(gain_count) / float(num_inputs) if num_inputs else 0.0,
                    "cumulative_covered_count": len(covered),
                    "cumulative_coverage": float(len(covered)) / float(num_inputs) if num_inputs else 0.0,
                    "covered_input_indices": "|".join(str(index) for index in sorted(new_covered)[:50]),
                }
            )
        orders[theta] = selected
    return orders, selected_rows


def evaluate_gt_fullgraph(
    records: list[MoleculeRecord],
    candidates: list[MoleculeRecord],
    *,
    top_k_list: list[int],
    theta_list: list[float],
    target_label: int,
    distance_proxy: DistanceProxy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_k = max(top_k_list) if top_k_list else 0
    orders, selected_rows = build_gt_greedy_orders(
        records,
        candidates,
        theta_list=theta_list,
        target_label=target_label,
        top_k_max=max_k,
        distance_proxy=distance_proxy,
    )

    per_input_rows: list[dict[str, Any]] = []
    for theta in theta_list:
        order = orders.get(theta, [])
        for k in top_k_list:
            prefix = order[: min(k, len(order))]
            for record in records:
                nearest: tuple[float, int, int, MoleculeRecord] | None = None
                for rank_position, candidate_index in enumerate(prefix, start=1):
                    candidate = candidates[candidate_index]
                    distance = distance_proxy.distance(record.canonical_smiles, candidate.canonical_smiles)
                    if not distance["ok"] or distance["cost"] is None:
                        continue
                    key = (float(distance["cost"]), rank_position)
                    if nearest is None or key < (nearest[0], nearest[1]):
                        nearest = (float(distance["cost"]), rank_position, candidate_index, candidate)
                if nearest is None:
                    choice = _choice_from_gt_candidate(
                        record,
                        None,
                        candidate_rank=None,
                        candidate_index=None,
                        target_label=target_label,
                        distance_proxy=distance_proxy,
                    )
                else:
                    _, rank_position, candidate_index, candidate = nearest
                    choice = _choice_from_gt_candidate(
                        record,
                        candidate,
                        candidate_rank=rank_position,
                        candidate_index=candidate_index,
                        target_label=target_label,
                        distance_proxy=distance_proxy,
                    )
                per_input_rows.append(
                    recourse_choice_to_row(
                        method="gt_fullgraph_greedy",
                        target_label=target_label,
                        k=k,
                        theta=theta,
                        record=record,
                        choice=choice,
                    )
                )
    return per_input_rows, selected_rows


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
    valid = [row for row in rows if row["valid_recourse"]]
    covered = [row for row in rows if row["covered"]]
    covered_costs = [float(row["cost"]) for row in covered if row.get("cost") not in (None, "")]
    valid_cf_drops = [float(row["cf_drop"]) for row in valid if row.get("cf_drop") not in (None, "")]
    return {
        "method": method,
        "target_label": int(target_label),
        "k": int(k),
        "theta": float(theta),
        "num_inputs": int(num_inputs),
        "coverage": float(len(covered)) / float(num_inputs) if num_inputs else 0.0,
        "median_cost": float(median(covered_costs)) if covered_costs else None,
        "mean_cost": _safe_mean(covered_costs),
        "mean_cf_drop": _safe_mean(valid_cf_drops),
        "flip_rate": float(sum(1 for row in valid if row["cf_flip"])) / float(num_inputs) if num_inputs else 0.0,
        "valid_recourse_rate": float(len(valid)) / float(num_inputs) if num_inputs else 0.0,
        "runtime_sec": float(runtime_sec),
    }


def build_comparison_table(
    ours_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    *,
    target_label: int,
    top_k_list: list[int],
    theta_list: list[float],
    num_inputs: int,
    runtime_sec: float,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    all_rows = {
        "ours_selected_subgraph": ours_rows,
        "gt_fullgraph_greedy": gt_rows,
    }
    for method, method_rows in all_rows.items():
        for k in top_k_list:
            for theta in theta_list:
                subset = [
                    row
                    for row in method_rows
                    if int(row["k"]) == int(k) and float(row["theta"]) == float(theta)
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start_time = time.time()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if Chem is None or rdFMCS is None:
        raise RuntimeError("RDKit is required for HIV quick recourse comparison.")

    top_k_list = sorted({int(value) for value in args.top_k_list if int(value) > 0})
    theta_list = sorted({float(value) for value in args.theta_list if float(value) >= 0.0})
    if not top_k_list:
        raise ValueError("--top-k-list must contain at least one positive integer.")
    if not theta_list:
        raise ValueError("--theta-list must contain at least one non-negative value.")

    teacher = RFTeacher(args.teacher_path)
    distance_proxy = DistanceProxy(timeout_sec=1)
    fragments, selected_files = load_selected_fragments(args.ours_selected_dir)
    max_requested_k = max(top_k_list)
    if len(fragments) < max_requested_k:
        print(
            f"[WARN] only {len(fragments)} selected fragments parsed, less than requested max k={max_requested_k}",
            file=sys.stderr,
        )

    target_records, gt_candidates, data_metadata = load_hiv_records(
        args.hiv_csv,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        target_label=int(args.target_label),
        teacher=teacher,
        max_inputs=args.max_inputs,
        max_gt_candidates=int(args.max_gt_candidates),
        seed=int(args.seed),
    )
    if not target_records:
        raise ValueError(f"No valid target-label inputs found for target_label={args.target_label}.")
    if not gt_candidates:
        raise ValueError(f"No valid opposite-label GT candidates found for target_label={args.target_label}.")

    ours_rows, ours_substructure_rates = evaluate_ours(
        target_records,
        fragments,
        top_k_list=top_k_list,
        theta_list=theta_list,
        target_label=int(args.target_label),
        teacher=teacher,
        distance_proxy=distance_proxy,
    )
    gt_rows, gt_selected_rows = evaluate_gt_fullgraph(
        target_records,
        gt_candidates,
        top_k_list=top_k_list,
        theta_list=theta_list,
        target_label=int(args.target_label),
        distance_proxy=distance_proxy,
    )

    runtime_sec = time.time() - start_time
    comparison_table = build_comparison_table(
        ours_rows,
        gt_rows,
        target_label=int(args.target_label),
        top_k_list=top_k_list,
        theta_list=theta_list,
        num_inputs=len(target_records),
        runtime_sec=runtime_sec,
    )

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
        "mcs_timeout_sec": 1,
    }
    summary = {
        "run_config": run_config,
        "data": data_metadata,
        "selected_files": selected_files,
        "selected_fragment_count": len(fragments),
        "selected_fragments": [
            {"rank": item.rank, "fragment": item.fragment, "source": item.source}
            for item in fragments
        ],
        "ours_substructure_match_rate_by_k": {
            str(k): value for k, value in sorted(ours_substructure_rates.items())
        },
        "comparison_table": comparison_table,
        "outputs": {
            "comparison_summary_json": str(out_dir / "comparison_summary.json"),
            "comparison_table_csv": str(out_dir / "comparison_table.csv"),
            "ours_per_input_csv": str(out_dir / "ours_per_input.csv"),
            "gt_fullgraph_per_input_csv": str(out_dir / "gt_fullgraph_per_input.csv"),
            "gt_selected_fullgraphs_csv": str(out_dir / "gt_selected_fullgraphs.csv"),
            "run_config_json": str(out_dir / "run_config.json"),
        },
    }

    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "comparison_summary.json", summary)
    write_csv(out_dir / "comparison_table.csv", comparison_table, COMPARISON_FIELDS)
    per_input_fields = [
        "method",
        "target_label",
        "k",
        "theta",
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
        "reason",
    ]
    write_csv(out_dir / "ours_per_input.csv", ours_rows, per_input_fields)
    write_csv(out_dir / "gt_fullgraph_per_input.csv", gt_rows, per_input_fields)
    write_csv(out_dir / "gt_selected_fullgraphs.csv", gt_selected_rows)

    print(f"[HIV_COMPARE] out_dir={out_dir}")
    print(f"[HIV_COMPARE] num_inputs={len(target_records)}")
    print(f"[HIV_COMPARE] gt_candidates={len(gt_candidates)}")
    print(f"[HIV_COMPARE] selected_fragments={len(fragments)}")
    print(f"[HIV_COMPARE] comparison_table={out_dir / 'comparison_table.csv'}")
    print(f"[HIV_COMPARE] comparison_summary={out_dir / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
