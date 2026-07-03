#!/usr/bin/env python3
"""Evaluate exported GlobalGCE outputs under the project AIDS/HIV CCRCov protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_adapter import (  # noqa: E402
    compute_globalgce_coverage_redundancy,
    compute_globalgce_structural_redundancy,
    globalgce_graph_record_to_mol,
    globalgce_rule_to_action,
    label_alignment_audit,
    load_globalgce_cfs,
    load_globalgce_rules,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - depends on runtime.
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


DEFAULT_THRESHOLDS = "0.05,0.10,0.20"
DEFAULT_TASK = "binary classification of HIV replication inhibition"
STRICT_CF_MODE = "strict_flip"
TANIMOTO_DISTANCE_TYPE = "tanimoto_fingerprint"


@dataclass
class ParentRecord:
    parent_id: str
    row_index: int
    smiles: str
    canonical_smiles: str | None
    label: int
    parse_ok: bool
    parse_error: str | None
    pred_label: int | None = None
    p_target: float | None = None
    teacher_ok: bool = False
    teacher_error: str | None = None


@dataclass
class CandidateRecord:
    candidate_id: str
    raw_index: int
    canonical_smiles: str
    source_smiles: str
    conversion_num_nodes: int
    conversion_num_edges: int
    graph_idx: Any | None
    cf_index: Any | None
    source_path: str | None
    pred_label: int | None = None
    p_target: float | None = None
    teacher_ok: bool = False
    teacher_error: str | None = None


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GlobalGCE exported CF graphs/rules under AIDS/HIV unified CCRCov.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-mode",
        choices=["native-cf", "native-cf-fullgraph", "native-cf-delta-action", "rule-action"],
        default="native-cf-fullgraph",
    )
    parser.add_argument("--run-root", default="outputs/hpc/globalgce/aids_official_top30")
    parser.add_argument("--export-dir", default="outputs/hpc/globalgce/aids_official_top30_exported")
    parser.add_argument("--dataset", default=None, help="Legacy alias for the official graph-baseline dataset key.")
    parser.add_argument("--dataset-key", default=None, help="Internal graph-baseline dataset key, normally aids.")
    parser.add_argument("--dataset-display-name", default=None, help="Paper-facing dataset name.")
    parser.add_argument("--dataset-csv", default=None, help="Canonical AIDS/HIV raw CSV.")
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--label", type=int, default=None, help="Legacy alias for --target-label.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--k", "--top-k", dest="top_k", type=int, default=30)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--teacher-path", default=None)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=None)
    parser.add_argument("--cfs-jsonl", default=None)
    parser.add_argument("--rules-jsonl", default=None)
    parser.add_argument("--distance-mode", choices=["tanimoto"], default="tanimoto")
    parser.add_argument("--cf-mode", choices=["strict_flip", "drop_or_flip", "drop_only"], default=STRICT_CF_MODE)
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--selection-policy", default="first_valid_unique_top_k")
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--config", default=None, help="Ignored compatibility hook for HPC wrappers.")
    parser.add_argument("--set", action="append", default=[], help="Ignored compatibility hook for HPC wrappers.")
    args = parser.parse_args()
    normalize_args(args)
    return args


def normalize_args(args: argparse.Namespace) -> None:
    dataset_key = args.dataset_key or _env_first("DATASET") or args.dataset or "aids"
    if str(dataset_key).strip().upper() == "AIDS":
        dataset_key = "aids"
    args.dataset_key = str(dataset_key).strip() or "aids"
    args.official_dataset = args.dataset or args.dataset_key
    args.dataset_display_name = (
        args.dataset_display_name
        or _env_first("DATASET_DISPLAY_NAME")
        or "AIDS/HIV"
    )
    args.dataset_csv = (
        args.dataset_csv
        or _env_first("AIDS_CSV", "HIV_CSV")
        or "data/raw/AIDS/HIV.csv"
    )
    args.smiles_col = args.smiles_col or _env_first("SMILES_COLUMN", "SMILES_COL") or "smiles"
    args.label_col = args.label_col or _env_first("LABEL_COLUMN", "LABEL_COL") or "HIV_active"
    args.target_label = int(
        args.target_label
        if args.target_label is not None
        else (_env_first("TARGET_LABEL", default=None) or (args.label if args.label is not None else 1))
    )
    args.teacher_path = args.teacher_path or _env_first("TEACHER_PATH")
    args.cfs_jsonl = args.cfs_jsonl or _env_first("CFS_JSONL") or str(resolve_path(args.export_dir) / "globalgce_cfs_graphs.jsonl")
    args.rules_jsonl = args.rules_jsonl or _env_first("RULES_JSONL") or str(resolve_path(args.export_dir) / "globalgce_rules.jsonl")
    if args.eval_mode == "native-cf":
        args.eval_mode = "native-cf-fullgraph"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path_like: str | Path) -> str:
    path = resolve_path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_label(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def safe_mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def safe_median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def canonicalize_smiles(smiles: str) -> tuple[str | None, str | None]:
    if Chem is None:
        return None, "rdkit_unavailable"
    normalized = str(smiles or "").strip()
    if not normalized:
        return None, "empty_smiles"
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None, "rdkit_parse_failed"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, f"sanitize_error:{exc}"
    return Chem.MolToSmiles(mol, canonical=True), None


def morgan_fingerprint(smiles: str) -> tuple[Any | None, str | None]:
    if Chem is None or DataStructs is None or rdFingerprintGenerator is None:
        return None, "rdkit_fingerprint_unavailable"
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None, "rdkit_parse_failed"
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol), None
    except Exception as exc:
        return None, f"fingerprint_failed:{exc}"


def tanimoto_distance(smiles_a: str, smiles_b: str) -> dict[str, Any]:
    fp_a, err_a = morgan_fingerprint(smiles_a)
    fp_b, err_b = morgan_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return {
            "distance": None,
            "similarity": None,
            "ok": False,
            "error": err_a or err_b or "fingerprint_failed",
        }
    similarity = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    return {
        "distance": float(1.0 - similarity),
        "similarity": similarity,
        "ok": True,
        "error": None,
    }


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return completed.stderr.strip() or "unknown"


def read_text_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def load_manifest_commit(export_dir: str | Path, run_root: str | Path) -> str | None:
    commit = read_text_if_exists(resolve_path(run_root) / "globalgce_source_commit.txt")
    if commit:
        return commit
    manifest_path = resolve_path(export_dir) / "globalgce_files_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return manifest.get("globalgce_commit")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.append({"raw_payload": payload, "line_number": line_number})
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def load_aids_hiv_parents(args: argparse.Namespace) -> tuple[list[ParentRecord], dict[str, Any]]:
    csv_path = resolve_path(args.dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing AIDS/HIV dataset CSV: {csv_path}")
    parents: list[ParentRecord] = []
    total_rows = 0
    target_rows = 0
    label_counts: Counter[str] = Counter()
    parse_ok_count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if args.smiles_col not in columns:
            raise ValueError(f"Missing SMILES column {args.smiles_col!r}; available={columns}")
        if args.label_col not in columns:
            raise ValueError(f"Missing label column {args.label_col!r}; available={columns}")
        for row_index, row in enumerate(reader):
            total_rows += 1
            label = parse_int_label(row.get(args.label_col))
            label_counts[str(label)] += 1
            if label != int(args.target_label):
                continue
            target_rows += 1
            if args.max_parents is not None and len(parents) >= int(args.max_parents):
                continue
            smiles = str(row.get(args.smiles_col) or "").strip()
            canonical, error = canonicalize_smiles(smiles)
            parse_ok = canonical is not None
            if parse_ok:
                parse_ok_count += 1
            parents.append(
                ParentRecord(
                    parent_id=str(row.get("id") or row.get("graph_id") or row_index),
                    row_index=row_index,
                    smiles=smiles,
                    canonical_smiles=canonical,
                    label=int(label),
                    parse_ok=parse_ok,
                    parse_error=error,
                )
            )
    audit = {
        "dataset_display_name": args.dataset_display_name,
        "dataset_key": args.dataset_key,
        "dataset_csv": display_path(args.dataset_csv),
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "target_label": int(args.target_label),
        "num_total": total_rows,
        "num_target_label": target_rows,
        "loaded_target_parents": len(parents),
        "label_distribution": dict(sorted(label_counts.items())),
        "parent_smiles_ok_rate": safe_rate(parse_ok_count, len(parents)),
    }
    return parents, audit


def load_teacher(teacher_path: str | Path | None) -> TeacherSemanticScorer:
    return TeacherSemanticScorer(teacher_path)


def predict_teacher_proba_smiles(
    teacher: TeacherSemanticScorer,
    smiles_list: Sequence[str],
    *,
    target_label: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for smiles in smiles_list:
        try:
            raw = teacher.score_smiles(smiles, label=int(target_label))
        except Exception as exc:
            results.append(
                {
                    "smiles": smiles,
                    "ok": False,
                    "pred_label": None,
                    "p_target": None,
                    "error": f"teacher_score_failed:{exc}",
                    "teacher_format": teacher.teacher_format,
                }
            )
            continue
        ok = bool(raw.get("teacher_result_ok"))
        results.append(
            {
                "smiles": smiles,
                "ok": ok,
                "pred_label": int(raw["teacher_label"]) if ok and raw.get("teacher_label") is not None else None,
                "p_target": float(raw["teacher_prob"]) if ok and raw.get("teacher_prob") is not None else None,
                "error": None if ok else str(raw.get("teacher_reason") or "teacher_result_not_ok"),
                "teacher_format": raw.get("teacher_format") or teacher.teacher_format,
            }
        )
    return results


def evaluate_parent_teacher(parents: list[ParentRecord], teacher: TeacherSemanticScorer, target_label: int) -> None:
    smiles = [parent.canonical_smiles or parent.smiles for parent in parents if parent.parse_ok]
    predictions = predict_teacher_proba_smiles(teacher, smiles, target_label=target_label)
    cursor = 0
    for parent in parents:
        if not parent.parse_ok:
            parent.teacher_ok = False
            parent.teacher_error = parent.parse_error
            continue
        pred = predictions[cursor]
        cursor += 1
        parent.teacher_ok = bool(pred.get("ok"))
        parent.pred_label = pred.get("pred_label")
        parent.p_target = pred.get("p_target")
        parent.teacher_error = pred.get("error")


def prepare_cf_candidates(
    args: argparse.Namespace,
    teacher: TeacherSemanticScorer,
    out_dir: Path,
) -> tuple[list[CandidateRecord], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfs = load_globalgce_cfs(resolve_path(args.cfs_jsonl))
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    candidates_by_smiles: dict[str, CandidateRecord] = {}
    invalid_reasons: Counter[str] = Counter()

    for raw_index, record in enumerate(cfs):
        conversion = globalgce_graph_record_to_mol(record)
        base = {
            "method": "GlobalGCE",
            "dataset": args.dataset_display_name,
            "dataset_key": args.dataset_key,
            "raw_index": raw_index,
            "cf_index": record.get("cf_index", raw_index),
            "graph_idx": record.get("graph_idx"),
            "source_path": record.get("source_path"),
            "conversion_ok": conversion.ok,
            "conversion_smiles": conversion.smiles,
            "conversion_num_nodes": conversion.num_nodes,
            "conversion_num_edges": conversion.num_edges,
            "error_type": conversion.error_type,
            "error_message": conversion.error_message,
            "invalid_reason": conversion.invalid_reason,
        }
        if not conversion.ok or not conversion.smiles:
            invalid_reasons[str(conversion.invalid_reason or conversion.error_type or "unknown")] += 1
            invalid_rows.append(base)
            continue
        if conversion.smiles in candidates_by_smiles:
            row = dict(base)
            row.update(
                {
                    "candidate_id": candidates_by_smiles[conversion.smiles].candidate_id,
                    "canonical_smiles": conversion.smiles,
                    "duplicate": True,
                    "selected": False,
                }
            )
            valid_rows.append(row)
            continue
        candidate_id = f"globalgce_cf_{len(candidates_by_smiles)}"
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            raw_index=raw_index,
            canonical_smiles=conversion.smiles,
            source_smiles=conversion.smiles,
            conversion_num_nodes=conversion.num_nodes,
            conversion_num_edges=conversion.num_edges,
            graph_idx=record.get("graph_idx"),
            cf_index=record.get("cf_index", raw_index),
            source_path=record.get("source_path"),
        )
        candidates_by_smiles[conversion.smiles] = candidate
        row = dict(base)
        row.update({"candidate_id": candidate_id, "canonical_smiles": conversion.smiles, "duplicate": False})
        valid_rows.append(row)

    unique_candidates = list(candidates_by_smiles.values())
    if args.max_candidates is not None:
        unique_candidates = unique_candidates[: int(args.max_candidates)]
    if int(args.top_k) > 0:
        selected_candidates = unique_candidates[: int(args.top_k)]
    else:
        selected_candidates = unique_candidates
    selected_ids = {candidate.candidate_id for candidate in selected_candidates}

    candidate_predictions = predict_teacher_proba_smiles(
        teacher,
        [candidate.canonical_smiles for candidate in unique_candidates],
        target_label=int(args.target_label),
    )
    for candidate, pred in zip(unique_candidates, candidate_predictions):
        candidate.teacher_ok = bool(pred.get("ok"))
        candidate.pred_label = pred.get("pred_label")
        candidate.p_target = pred.get("p_target")
        candidate.teacher_error = pred.get("error")

    for candidate in unique_candidates:
        for row in [item for item in valid_rows if item.get("candidate_id") == candidate.candidate_id]:
            row.update(
                {
                    "selected": candidate.candidate_id in selected_ids,
                    "teacher_ok": candidate.teacher_ok,
                    "teacher_pred_label": candidate.pred_label,
                    "teacher_p_target": candidate.p_target,
                    "teacher_error": candidate.teacher_error,
                }
            )

    selected_candidates = [candidate for candidate in selected_candidates if candidate.teacher_ok]
    stats = {
        "num_cfs_total": len(cfs),
        "num_candidates_raw": len(cfs),
        "num_candidates_before_topk": len(unique_candidates),
        "num_candidates_after_topk": len(selected_ids),
        "num_candidates_valid": len(selected_candidates),
        "cfs_conversion_ok": sum(1 for row in valid_rows if not row.get("duplicate")),
        "cfs_conversion_fail": len(invalid_rows),
        "cfs_smiles_ok_rate": safe_rate(sum(1 for row in valid_rows if not row.get("duplicate")), len(cfs)),
        "cfs_teacher_ok": sum(1 for candidate in unique_candidates if candidate.teacher_ok),
        "cfs_teacher_ok_rate": safe_rate(sum(1 for candidate in unique_candidates if candidate.teacher_ok), len(unique_candidates)),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "selection_policy": args.selection_policy,
    }
    write_csv(
        out_dir / "valid_candidates.csv",
        valid_rows,
        [
            "method",
            "dataset",
            "dataset_key",
            "raw_index",
            "candidate_id",
            "cf_index",
            "graph_idx",
            "canonical_smiles",
            "conversion_ok",
            "conversion_num_nodes",
            "conversion_num_edges",
            "duplicate",
            "selected",
            "teacher_ok",
            "teacher_pred_label",
            "teacher_p_target",
            "teacher_error",
            "source_path",
        ],
    )
    write_csv(
        out_dir / "invalid_candidates.csv",
        invalid_rows,
        [
            "method",
            "dataset",
            "dataset_key",
            "raw_index",
            "cf_index",
            "graph_idx",
            "conversion_ok",
            "conversion_smiles",
            "conversion_num_nodes",
            "conversion_num_edges",
            "error_type",
            "error_message",
            "invalid_reason",
            "source_path",
        ],
    )
    return selected_candidates, stats, valid_rows, invalid_rows


def cf_condition(pred_after: int | None, cf_drop: float | None, *, target_label: int, cf_mode: str, min_cf_drop: float) -> bool:
    strict_flip = pred_after is not None and int(pred_after) != int(target_label)
    drop_ok = cf_drop is not None and float(cf_drop) >= float(min_cf_drop)
    if cf_mode == "strict_flip":
        return strict_flip
    if cf_mode == "drop_or_flip":
        return strict_flip or drop_ok
    if cf_mode == "drop_only":
        return drop_ok
    return strict_flip


def build_details(
    parents: list[ParentRecord],
    candidates: list[CandidateRecord],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent in parents:
        if not parent.parse_ok or not parent.teacher_ok:
            continue
        for candidate in candidates:
            distance = tanimoto_distance(parent.canonical_smiles or parent.smiles, candidate.canonical_smiles)
            p_after = candidate.p_target
            cf_drop = (
                float(parent.p_target) - float(p_after)
                if parent.p_target is not None and p_after is not None
                else None
            )
            cf_flip = candidate.pred_label is not None and int(candidate.pred_label) != int(args.target_label)
            rows.append(
                {
                    "method": "GlobalGCE",
                    "dataset": args.dataset_display_name,
                    "dataset_key": args.dataset_key,
                    "eval_mode": args.eval_mode,
                    "cf_mode": args.cf_mode,
                    "distance_mode": args.distance_mode,
                    "distance_type": TANIMOTO_DISTANCE_TYPE,
                    "parent_id": parent.parent_id,
                    "parent_row_index": parent.row_index,
                    "parent_smiles": parent.smiles,
                    "parent_canonical_smiles": parent.canonical_smiles,
                    "target_label": int(args.target_label),
                    "candidate_id": candidate.candidate_id,
                    "candidate_smiles": candidate.canonical_smiles,
                    "candidate_cf_index": candidate.cf_index,
                    "candidate_graph_idx": candidate.graph_idx,
                    "p_before": parent.p_target,
                    "p_after": p_after,
                    "pred_before": parent.pred_label,
                    "pred_after": candidate.pred_label,
                    "cf_drop": cf_drop,
                    "cf_flip": cf_flip,
                    "distance": distance.get("distance"),
                    "tanimoto_similarity": distance.get("similarity"),
                    "distance_ok": distance.get("ok"),
                    "error": distance.get("error") or candidate.teacher_error or parent.teacher_error,
                }
            )
    return rows


def summarize_thresholds(
    details: list[dict[str, Any]],
    parents: list[ParentRecord],
    candidates: list[CandidateRecord],
    thresholds: Sequence[float],
    args: argparse.Namespace,
    candidate_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_parent_count = len(parents)
    for threshold in thresholds:
        best_by_parent: dict[str, dict[str, Any]] = {}
        close_only: set[str] = set()
        for detail in details:
            distance = detail.get("distance")
            if distance in (None, ""):
                continue
            if float(distance) > float(threshold):
                continue
            parent_id = str(detail.get("parent_id"))
            close_only.add(parent_id)
            if not cf_condition(
                detail.get("pred_after"),
                detail.get("cf_drop"),
                target_label=int(args.target_label),
                cf_mode=args.cf_mode,
                min_cf_drop=float(args.min_cf_drop),
            ):
                continue
            current = best_by_parent.get(parent_id)
            if current is None:
                best_by_parent[parent_id] = detail
                continue
            current_key = (float(current.get("distance") or 999.0), -float(current.get("cf_drop") or 0.0))
            candidate_key = (float(detail.get("distance") or 999.0), -float(detail.get("cf_drop") or 0.0))
            if candidate_key < current_key:
                best_by_parent[parent_id] = detail
        covered = list(best_by_parent.values())
        best_distances = [float(row["distance"]) for row in covered if row.get("distance") not in (None, "")]
        cf_drops = [float(row["cf_drop"]) for row in covered if row.get("cf_drop") not in (None, "")]
        flips = [bool(row.get("cf_flip")) for row in covered]
        rows.append(
            {
                "method": "GlobalGCE",
                "dataset": args.dataset_display_name,
                "dataset_key": args.dataset_key,
                "raw_source": display_path(args.dataset_csv),
                "eval_mode": args.eval_mode,
                "threshold": float(threshold),
                "target_label": int(args.target_label),
                "cf_mode": args.cf_mode,
                "min_cf_drop": float(args.min_cf_drop),
                "distance_mode": args.distance_mode,
                "distance_type": TANIMOTO_DISTANCE_TYPE,
                "num_parents": total_parent_count,
                "num_candidates": len(candidates),
                "num_valid_candidates": candidate_stats.get("num_candidates_valid"),
                "num_close_only_covered": len(close_only),
                "close_only_coverage": safe_rate(len(close_only), total_parent_count),
                "num_close_cf_covered": len(covered),
                "close_cf_coverage": safe_rate(len(covered), total_parent_count),
                "CCRCov": safe_rate(len(covered), total_parent_count),
                "avg_best_distance": safe_mean(best_distances),
                "median_best_distance": safe_median(best_distances),
                "avg_cf_drop_among_covered": safe_mean(cf_drops),
                "flip_rate_among_covered": safe_rate(sum(1 for item in flips if item), len(flips)),
                "valid_rate": safe_rate(int(candidate_stats.get("num_candidates_valid") or 0), int(candidate_stats.get("num_cfs_total") or 0)),
                "teacher_eval_success_rate": None,
                "suppcov_status": "skipped_for_fullgraph_native_cf",
            }
        )
    return rows


def evaluate_native_cf_fullgraph(args: argparse.Namespace, thresholds: list[float], out_dir: Path) -> dict[str, Any]:
    if not args.teacher_path:
        raise ValueError("--teacher-path is required for GlobalGCE native-cf-fullgraph evaluation")
    parents, dataset_audit = load_aids_hiv_parents(args)
    teacher = load_teacher(args.teacher_path)
    evaluate_parent_teacher(parents, teacher, int(args.target_label))
    selected_candidates, candidate_stats, _valid_rows, _invalid_rows = prepare_cf_candidates(args, teacher, out_dir)
    details = build_details(parents, selected_candidates, args)
    summary_rows = summarize_thresholds(details, parents, selected_candidates, thresholds, args, candidate_stats)

    parent_teacher_ok = sum(1 for parent in parents if parent.teacher_ok)
    teacher_attempts = len(parents) + int(candidate_stats.get("num_candidates_before_topk") or 0)
    teacher_ok = parent_teacher_ok + int(candidate_stats.get("cfs_teacher_ok") or 0)
    teacher_success_rate = safe_rate(teacher_ok, teacher_attempts)
    for row in summary_rows:
        row["teacher_eval_success_rate"] = teacher_success_rate

    write_csv(out_dir / "details.csv", details, DETAILS_FIELDS)
    write_csv(out_dir / "threshold_summary.csv", summary_rows, THRESHOLD_FIELDS)

    key_threshold = 0.20 if any(abs(float(item) - 0.20) < 1e-9 for item in thresholds) else (max(thresholds) if thresholds else 0.20)
    key_row = next((row for row in summary_rows if abs(float(row["threshold"]) - float(key_threshold)) < 1e-9), summary_rows[-1] if summary_rows else {})
    summary = {
        "method": "GlobalGCE",
        "dataset": args.dataset_display_name,
        "dataset_key": args.dataset_key,
        "raw_source": display_path(args.dataset_csv),
        "task": args.task,
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "eval_mode": args.eval_mode,
        "target_label": int(args.target_label),
        "cf_mode": args.cf_mode,
        "distance_mode": args.distance_mode,
        "distance_type": TANIMOTO_DISTANCE_TYPE,
        "top_k": int(args.top_k),
        "selection_policy": args.selection_policy,
        "num_parents": len(parents),
        "num_cfs_total": candidate_stats.get("num_cfs_total"),
        "num_candidates_raw": candidate_stats.get("num_candidates_raw"),
        "num_candidates_valid": candidate_stats.get("num_candidates_valid"),
        "num_candidates_before_topk": candidate_stats.get("num_candidates_before_topk"),
        "num_candidates_after_topk": candidate_stats.get("num_candidates_after_topk"),
        "cfs_conversion_ok": candidate_stats.get("cfs_conversion_ok"),
        "cfs_conversion_fail": candidate_stats.get("cfs_conversion_fail"),
        "valid_rate": safe_rate(int(candidate_stats.get("num_candidates_valid") or 0), int(candidate_stats.get("num_cfs_total") or 0)),
        "cfs_smiles_ok_rate": candidate_stats.get("cfs_smiles_ok_rate"),
        "cfs_teacher_ok": candidate_stats.get("cfs_teacher_ok"),
        "cfs_teacher_ok_rate": candidate_stats.get("cfs_teacher_ok_rate"),
        "parent_smiles_ok_rate": dataset_audit.get("parent_smiles_ok_rate"),
        "parent_teacher_ok": parent_teacher_ok,
        "teacher_load_ok": bool(teacher.available),
        "teacher_input_type": teacher.teacher_format,
        "teacher_path": str(resolve_path(args.teacher_path)),
        "teacher_eval_success_rate": teacher_success_rate,
        "first_5_parent_predictions": [
            {
                "parent_id": parent.parent_id,
                "smiles": parent.smiles,
                "pred_label": parent.pred_label,
                "p_target": parent.p_target,
                "teacher_ok": parent.teacher_ok,
                "teacher_error": parent.teacher_error,
            }
            for parent in parents[:5]
        ],
        "CCRCov@0.05": _summary_value(summary_rows, 0.05, "close_cf_coverage"),
        "CCRCov@0.10": _summary_value(summary_rows, 0.10, "close_cf_coverage"),
        "CCRCov@0.20": _summary_value(summary_rows, 0.20, "close_cf_coverage"),
        "CFDrop": key_row.get("avg_cf_drop_among_covered"),
        "FlipRate": key_row.get("flip_rate_among_covered"),
        "CostMean": key_row.get("avg_best_distance"),
        "CostMedian": key_row.get("median_best_distance"),
        "SuppCov@K": None,
        "suppcov_status": "skipped_for_fullgraph_native_cf",
        "StructRed": None,
        "CovRed": None,
        "invalid_reason_counts": candidate_stats.get("invalid_reason_counts"),
        "label_alignment_warning": label_alignment_audit()["label_alignment_warning"],
        "source_run_root": str(resolve_path(args.run_root)),
        "source_export_dir": str(resolve_path(args.export_dir)),
        "cfs_jsonl": str(resolve_path(args.cfs_jsonl)),
        "rules_jsonl": str(resolve_path(args.rules_jsonl)),
        "globalgce_commit": load_manifest_commit(args.export_dir, args.run_root),
        "project_git_commit": git_commit(),
    }
    audit = {
        **dataset_audit,
        "teacher_load_ok": bool(teacher.available),
        "teacher_input_type": teacher.teacher_format,
        "teacher_availability_reason": teacher.availability_reason,
        "first_5_parent_predictions": summary["first_5_parent_predictions"],
        "candidate_conversion": candidate_stats,
        "thresholds": thresholds,
        "LABEL_ALIGNMENT_AUDIT": {
            "globalgce_official_dataset": "AIDS graph format",
            "project_dataset": args.dataset_display_name,
            "project_label_col": args.label_col,
            "target_label": int(args.target_label),
            "warning": (
                "GlobalGCE official preprocessing may flip AIDS internal graph labels; do not equate "
                "GlobalGCE internal label=1 with project HIV_active=1."
            ),
            "current_eval_interpretation": (
                "This evaluation treats exported GlobalGCE CF graphs as baseline-generated candidate molecules "
                "and evaluates them against project HIV_active labels using the project teacher."
            ),
        },
    }
    write_json(out_dir / "audit.json", audit)
    return summary


def _summary_value(rows: list[dict[str, Any]], threshold: float, key: str) -> Any:
    for row in rows:
        if abs(float(row.get("threshold") or -1.0) - float(threshold)) < 1e-9:
            return row.get(key)
    return None


def evaluate_native_cf_delta_action(args: argparse.Namespace, thresholds: list[float], out_dir: Path) -> dict[str, Any]:
    del thresholds
    cfs = load_globalgce_cfs(resolve_path(args.cfs_jsonl))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(cfs):
        rows.append(
            {
                "cf_index": record.get("cf_index", index),
                "graph_idx": record.get("graph_idx"),
                "projected_delta_supported": False,
                "projected_delta_reason": "missing_source_parent_mapping_or_atom_mapping",
                "note": "GlobalGCE export lacks reliable source-parent atom mapping for safe delta extraction.",
            }
        )
    write_jsonl(
        out_dir / "projected_delta_actions.jsonl",
        rows,
    )
    summary = {
        "method": "GlobalGCE",
        "dataset": args.dataset_display_name,
        "dataset_key": args.dataset_key,
        "raw_source": display_path(args.dataset_csv),
        "task": args.task,
        "label_col": args.label_col,
        "eval_mode": args.eval_mode,
        "target_label": int(args.target_label),
        "cf_mode": args.cf_mode,
        "distance_mode": args.distance_mode,
        "distance_type": TANIMOTO_DISTANCE_TYPE,
        "top_k": int(args.top_k),
        "num_cfs_total": len(cfs),
        "projected_delta_supported": False,
        "projected_delta_reason": "missing_source_parent_mapping_or_atom_mapping",
        "SuppCov@K": None,
        "CFDrop": None,
        "FlipRate": None,
        "CostMean": None,
        "CostMedian": None,
        "StructRed": None,
        "CovRed": None,
        "valid_rate": None,
        "label_alignment_warning": label_alignment_audit()["label_alignment_warning"],
        "project_git_commit": git_commit(),
    }
    write_csv(out_dir / "details.csv", rows, ["cf_index", "graph_idx", "projected_delta_supported", "projected_delta_reason", "note"])
    write_csv(out_dir / "threshold_summary.csv", [], THRESHOLD_FIELDS)
    write_csv(out_dir / "valid_candidates.csv", [], ["candidate_id", "canonical_smiles"])
    write_csv(out_dir / "invalid_candidates.csv", [], ["candidate_id", "invalid_reason"])
    write_json(out_dir / "audit.json", {"LABEL_ALIGNMENT_AUDIT": label_alignment_audit(), "projected_delta_actions": summary})
    return summary


def evaluate_rule_action(args: argparse.Namespace, thresholds: list[float], out_dir: Path) -> dict[str, Any]:
    parents, dataset_audit = load_aids_hiv_parents(args)
    rules = load_globalgce_rules(resolve_path(args.rules_jsonl))
    if int(args.top_k) > 0:
        rules = rules[: int(args.top_k)]
    rule_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    cover_sets: dict[int, set[str]] = {}
    for rule_index, rule in enumerate(rules):
        rule_id = int(rule.get("rule_id", rule_index))
        lhs_conversion = globalgce_graph_record_to_mol(rule.get("lhs") or {})
        rhs_conversion = globalgce_graph_record_to_mol(rule.get("rhs") or {})
        matched: set[str] = set()
        unsupported_reason = "unsafe_lhs_rhs_replacement_without_attachment_mapping"
        if Chem is None:
            unsupported_reason = "rdkit_unavailable"
        elif lhs_conversion.ok and lhs_conversion.mol is not None:
            for parent in parents:
                if not parent.parse_ok:
                    continue
                parent_mol = Chem.MolFromSmiles(parent.canonical_smiles or parent.smiles)
                if parent_mol is not None and parent_mol.HasSubstructMatch(lhs_conversion.mol):
                    matched.add(parent.parent_id)
                    detail_rows.append(
                        {
                            "method": "GlobalGCE",
                            "dataset": args.dataset_display_name,
                            "dataset_key": args.dataset_key,
                            "eval_mode": args.eval_mode,
                            "parent_id": parent.parent_id,
                            "parent_smiles": parent.smiles,
                            "rule_id": rule_id,
                            "lhs_smiles": lhs_conversion.smiles,
                            "rhs_smiles": rhs_conversion.smiles,
                            "suppcov_match": True,
                            "rule_action_supported": False,
                            "unsupported_reason": unsupported_reason,
                        }
                    )
        cover_sets[rule_id] = matched
        action = globalgce_rule_to_action(rule)
        rule_rows.append(
            {
                "rule_id": rule_id,
                "lhs_smiles": lhs_conversion.smiles,
                "rhs_smiles": rhs_conversion.smiles,
                "lhs_conversion_ok": lhs_conversion.ok,
                "rhs_conversion_ok": rhs_conversion.ok,
                "lhs_invalid_reason": lhs_conversion.invalid_reason,
                "rhs_invalid_reason": rhs_conversion.invalid_reason,
                "match_count": len(matched),
                "covered_parent_ids": sorted(matched),
                "rule_action_supported": False,
                "unsupported_reason": unsupported_reason,
                "adapter_note": action.get("unsupported_reason"),
            }
        )
    covered_by_any: set[str] = set()
    for covered in cover_sets.values():
        covered_by_any.update(covered)
    summary = {
        "method": "GlobalGCE",
        "dataset": args.dataset_display_name,
        "dataset_key": args.dataset_key,
        "raw_source": display_path(args.dataset_csv),
        "task": args.task,
        "label_col": args.label_col,
        "eval_mode": args.eval_mode,
        "target_label": int(args.target_label),
        "cf_mode": args.cf_mode,
        "distance_mode": args.distance_mode,
        "distance_type": TANIMOTO_DISTANCE_TYPE,
        "top_k": int(args.top_k),
        "num_parents": len(parents),
        "num_rules": len(rules),
        "num_cfs_total": len(read_jsonl(args.cfs_jsonl)) if resolve_path(args.cfs_jsonl).exists() else None,
        "SuppCov@K": safe_rate(len(covered_by_any), len(parents)),
        "CFDrop": None,
        "FlipRate": None,
        "CostMean": None,
        "CostMedian": None,
        "StructRed": compute_globalgce_structural_redundancy(rules),
        "CovRed": compute_globalgce_coverage_redundancy(cover_sets),
        "valid_rate": None,
        "rule_action_supported": False,
        "unsupported_reason": "unsafe_lhs_rhs_replacement_without_attachment_mapping",
        "label_alignment_warning": label_alignment_audit()["label_alignment_warning"],
        "project_git_commit": git_commit(),
    }
    for threshold in thresholds:
        summary[f"CCRCov@{threshold:.2f}"] = None
    write_csv(
        out_dir / "per_rule.csv",
        rule_rows,
        [
            "rule_id",
            "lhs_smiles",
            "rhs_smiles",
            "lhs_conversion_ok",
            "rhs_conversion_ok",
            "lhs_invalid_reason",
            "rhs_invalid_reason",
            "match_count",
            "covered_parent_ids",
            "rule_action_supported",
            "unsupported_reason",
            "adapter_note",
        ],
    )
    write_csv(
        out_dir / "details.csv",
        detail_rows,
        [
            "method",
            "dataset",
            "dataset_key",
            "eval_mode",
            "parent_id",
            "parent_smiles",
            "rule_id",
            "lhs_smiles",
            "rhs_smiles",
            "suppcov_match",
            "rule_action_supported",
            "unsupported_reason",
        ],
    )
    write_csv(out_dir / "threshold_summary.csv", [], THRESHOLD_FIELDS)
    write_csv(out_dir / "valid_candidates.csv", [], ["candidate_id", "canonical_smiles"])
    write_csv(out_dir / "invalid_candidates.csv", [], ["candidate_id", "invalid_reason"])
    write_json(
        out_dir / "audit.json",
        {
            **dataset_audit,
            "LABEL_ALIGNMENT_AUDIT": label_alignment_audit(),
            "rule_action_supported": False,
            "unsupported_reason": "unsafe_lhs_rhs_replacement_without_attachment_mapping",
        },
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    no_covered = not summary.get("CCRCov@0.05") and not summary.get("CCRCov@0.10") and not summary.get("CCRCov@0.20")
    lines = [
        "GlobalGCE AIDS/HIV Unified Evaluation Report",
        "",
        f"Method: {summary.get('method')}",
        f"Dataset: {summary.get('dataset')}",
        f"Raw source: {summary.get('raw_source')}",
        f"Task: {summary.get('task')}",
        f"Label: {summary.get('label_col')}",
        f"Target label: {summary.get('target_label')}",
        f"CLEAR / official graph baseline key: {summary.get('dataset_key')}",
        "Ours legacy script name: hiv",
        "Unified project dataset: AIDS/HIV",
        "",
        f"Eval mode: {summary.get('eval_mode')}",
        f"CF mode: {summary.get('cf_mode')}",
        f"Distance mode: {summary.get('distance_mode')}",
        f"Distance type: {summary.get('distance_type')}",
        f"Top K: {summary.get('top_k')}",
        "",
        f"CCRCov@0.05: {summary.get('CCRCov@0.05')}",
        f"CCRCov@0.10: {summary.get('CCRCov@0.10')}",
        f"CCRCov@0.20: {summary.get('CCRCov@0.20')}",
        f"CFDrop: {summary.get('CFDrop')}",
        f"FlipRate: {summary.get('FlipRate')}",
        f"CostMean: {summary.get('CostMean')}",
        f"CostMedian: {summary.get('CostMedian')}",
        f"ValidRate: {summary.get('valid_rate')}",
        f"teacher_eval_success_rate: {summary.get('teacher_eval_success_rate')}",
        f"SuppCov@K: {summary.get('SuppCov@K')}",
        f"suppcov_status: {summary.get('suppcov_status')}",
        "",
        "Interpretation:",
        "native-cf-fullgraph treats converted GlobalGCE CF graphs as complete counterfactual molecule candidates.",
        "SuppCov is skipped for this mode because fullgraph candidates are not LHS/RHS support rules.",
        "If distance_type=tanimoto_fingerprint, this is a smoke/diagnostic distance and not the final GREED-GED line.",
        "GlobalGCE official AIDS graph/export artifacts are baseline raw outputs; final labels come from HIV_active in the AIDS/HIV CSV.",
        "",
        "Label alignment warning:",
        str(summary.get("label_alignment_warning")),
    ]
    if no_covered and summary.get("eval_mode") == "native-cf-fullgraph":
        lines.extend(
            [
                "",
                "No close-CF covered samples under current thresholds.",
                "Therefore CFDrop and Cost may be null because no parent had a candidate satisfying both distance and strict flip conditions.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DETAILS_FIELDS = [
    "method",
    "dataset",
    "dataset_key",
    "eval_mode",
    "cf_mode",
    "distance_mode",
    "distance_type",
    "parent_id",
    "parent_row_index",
    "parent_smiles",
    "parent_canonical_smiles",
    "target_label",
    "candidate_id",
    "candidate_smiles",
    "candidate_cf_index",
    "candidate_graph_idx",
    "p_before",
    "p_after",
    "pred_before",
    "pred_after",
    "cf_drop",
    "cf_flip",
    "distance",
    "tanimoto_similarity",
    "distance_ok",
    "error",
]

THRESHOLD_FIELDS = [
    "method",
    "dataset",
    "dataset_key",
    "raw_source",
    "eval_mode",
    "threshold",
    "target_label",
    "cf_mode",
    "min_cf_drop",
    "distance_mode",
    "distance_type",
    "num_parents",
    "num_candidates",
    "num_valid_candidates",
    "num_close_only_covered",
    "close_only_coverage",
    "num_close_cf_covered",
    "close_cf_coverage",
    "CCRCov",
    "avg_best_distance",
    "median_best_distance",
    "avg_cf_drop_among_covered",
    "flip_rate_among_covered",
    "valid_rate",
    "teacher_eval_success_rate",
    "suppcov_status",
]


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if args.output_dir:
        out_dir = resolve_path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "outputs" / "hpc" / "eval" / "globalgce" / "aids_hiv_native_cf_fullgraph" / f"label{args.target_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[GLOBALGCE_EVAL_CONFIG]")
    print(f"dataset_display_name={args.dataset_display_name}")
    print(f"dataset_key={args.dataset_key}")
    print(f"dataset_csv={resolve_path(args.dataset_csv)}")
    print(f"smiles_col={args.smiles_col}")
    print(f"label_col={args.label_col}")
    print(f"target_label={args.target_label}")
    print(f"eval_mode={args.eval_mode}")
    print(f"cf_mode={args.cf_mode}")
    print(f"distance_type={TANIMOTO_DISTANCE_TYPE}")
    print(f"output_dir={out_dir}")

    if args.eval_mode == "native-cf-fullgraph":
        summary = evaluate_native_cf_fullgraph(args, thresholds, out_dir)
    elif args.eval_mode == "native-cf-delta-action":
        summary = evaluate_native_cf_delta_action(args, thresholds, out_dir)
    else:
        summary = evaluate_rule_action(args, thresholds, out_dir)

    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.txt", summary)
    print("[GLOBALGCE_EVAL_DONE]")
    print(f"output_dir={out_dir}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
