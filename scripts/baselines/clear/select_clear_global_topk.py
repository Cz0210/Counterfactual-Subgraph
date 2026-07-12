#!/usr/bin/env python3
"""Select a global CLEAR full-molecule Top-K set before expensive evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chem import parse_smiles  # noqa: E402
from src.eval.class_counterfactual_selector import compute_weighted_mmr_score  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


SELECTOR_NAME = "CLEAR-RF-StrictFlip-Greedy-MMR"
SELECTOR_VERSION = "clear_global_topk_morgan_v1"
DEFAULT_REFERENCE_SUMMARY = (
    "outputs/hpc/selectors/"
    "stable300_label1_merged_base_temp07_top20_mmr_cov20/selector_summary.json"
)
OUTPUT_FIELDS = [
    "rank",
    "candidate_id",
    "candidate_smiles",
    "canonical_smiles",
    "source_method",
    "source_frequency",
    "rf_pred_label",
    "rf_prob_label0",
    "rf_prob_label1",
    "rf_strict_flip",
    "official_target_success",
    "total_action_cost",
    "normalized_cost",
    "coverage_count",
    "new_coverage_count",
    "marginal_gain",
    "redundancy_penalty",
    "source_exp_id",
    "source_instance_index",
]
TRACE_FIELDS = OUTPUT_FIELDS + [
    "coverage_gain",
    "rf_counterfactual_confidence",
    "weighted_cf_component",
    "weighted_coverage_component",
    "weighted_cost_penalty",
    "weighted_redundancy_penalty",
    "cumulative_coverage_count",
    "cumulative_coverage_rate",
]
_POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass
class ClearCandidate:
    candidate_id: str
    candidate_smiles: str
    canonical_smiles: str
    source_method: str
    source_order: int
    source_frequency: int = 1
    official_target_success: bool = False
    total_action_cost: float | None = None
    source_exp_id: str = ""
    source_instance_index: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    rf_pred_label: int | None = None
    rf_prob_label0: float | None = None
    rf_prob_label1: float | None = None
    rf_strict_flip: bool = False
    rf_counterfactual_confidence: float = 0.0
    normalized_cost: float = 1.0
    coverage_bits: np.ndarray | None = None
    coverage_count: int = 0
    fingerprint: Any = None


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def _as_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "t", "yes", "y", "on"}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_weights(args: argparse.Namespace) -> tuple[dict[str, float], str]:
    explicit = {
        "w_cov": args.w_cov,
        "w_cf": args.w_cf,
        "w_cost": args.w_cost,
        "w_red": args.w_red,
    }
    if all(value is not None for value in explicit.values()):
        return {key: float(value) for key, value in explicit.items()}, "explicit_cli"

    reference_path = Path(args.reference_selector_summary).expanduser().resolve()
    metadata: dict[str, Any] = {}
    if reference_path.is_file():
        try:
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        except Exception as exc:
            raise ValueError(f"Could not read reference selector summary {reference_path}: {exc}") from exc
    mapped = {
        "w_cov": _as_float(metadata.get("beta_coverage")),
        "w_cf": _as_float(metadata.get("alpha_cf")),
        "w_cost": _as_float(metadata.get("eta_size")),
        "w_red": _as_float(metadata.get("gamma_redundancy")),
    }
    for key, value in explicit.items():
        if value is not None:
            mapped[key] = float(value)
    missing = [key for key, value in mapped.items() if value is None]
    if missing:
        raise ValueError(
            "Selector weights are not fully known. Provide --w-cov, --w-cf, --w-cost, and --w-red, "
            f"or provide a valid --reference-selector-summary. Missing={missing}; reference={reference_path}"
        )
    return {key: float(value) for key, value in mapped.items()}, f"reference_selector_summary:{reference_path}"


def _candidate_is_better_representative(
    *,
    new_cost: float | None,
    new_order: int,
    current: ClearCandidate,
) -> bool:
    current_cost = current.total_action_cost
    if new_cost is not None and current_cost is None:
        return True
    if new_cost is None and current_cost is not None:
        return False
    if new_cost is not None and current_cost is not None and not math.isclose(new_cost, current_cost):
        return new_cost < current_cost
    return new_order < current.source_order


def _load_and_deduplicate_candidates(
    path: Path,
    *,
    smiles_col: str,
) -> tuple[list[ClearCandidate], dict[str, Any]]:
    counters: Counter[str] = Counter()
    by_canonical: dict[str, ClearCandidate] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if smiles_col not in (reader.fieldnames or []):
            raise ValueError(f"Candidate CSV missing {smiles_col!r}; columns={reader.fieldnames or []}")
        for row_index, row in enumerate(reader):
            counters["candidate_count_before_filter"] += 1
            if not _as_bool(row.get("candidate_valid")):
                counters["candidate_valid_false"] += 1
                continue
            smiles = _text(row.get(smiles_col))
            if not smiles:
                counters["candidate_smiles_missing"] += 1
                continue
            parsed = parse_smiles(smiles, sanitize=True, canonicalize=True, allow_capped_fragments=False)
            if not parsed.parseable or not parsed.sanitized or parsed.mol is None or not parsed.canonical_smiles:
                counters["candidate_rdkit_invalid"] += 1
                continue
            canonical = str(parsed.canonical_smiles)
            cost = _as_float(row.get("total_action_cost"))
            candidate_id = _text(row.get("candidate_id")) or f"clear_candidate_{row_index}"
            existing = by_canonical.get(canonical)
            if existing is None:
                by_canonical[canonical] = ClearCandidate(
                    candidate_id=candidate_id,
                    candidate_smiles=canonical,
                    canonical_smiles=canonical,
                    source_method=_text(row.get("source_method")) or "CLEAR",
                    source_order=row_index,
                    official_target_success=_as_bool(row.get("official_target_success")),
                    total_action_cost=cost,
                    source_exp_id=_text(row.get("source_exp_id")),
                    source_instance_index=_text(row.get("source_instance_index")),
                    raw=dict(row),
                )
                continue
            existing.source_frequency += 1
            existing.official_target_success = bool(
                existing.official_target_success or _as_bool(row.get("official_target_success"))
            )
            if _candidate_is_better_representative(new_cost=cost, new_order=row_index, current=existing):
                existing.candidate_id = candidate_id
                existing.candidate_smiles = canonical
                existing.source_method = _text(row.get("source_method")) or "CLEAR"
                existing.source_order = row_index
                existing.total_action_cost = cost
                existing.source_exp_id = _text(row.get("source_exp_id"))
                existing.source_instance_index = _text(row.get("source_instance_index"))
                existing.raw = dict(row)
    counters["unique_candidate_count"] = len(by_canonical)
    return list(by_canonical.values()), dict(counters)


def _load_parents(
    path: Path,
    *,
    smiles_col: str,
    label_col: str,
    target_label: int,
) -> tuple[list[str], dict[str, Any]]:
    canonical_smiles: list[str] = []
    counters: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in (smiles_col, label_col) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Parent CSV missing columns={missing}; available={reader.fieldnames or []}")
        for row in reader:
            counters["parent_rows_total"] += 1
            if _as_int(row.get(label_col)) != int(target_label):
                counters["parent_label_filtered"] += 1
                continue
            parsed = parse_smiles(
                _text(row.get(smiles_col)),
                sanitize=True,
                canonicalize=True,
                allow_capped_fragments=False,
            )
            if not parsed.parseable or not parsed.sanitized or parsed.mol is None or not parsed.canonical_smiles:
                counters["parent_rdkit_invalid"] += 1
                continue
            canonical_smiles.append(str(parsed.canonical_smiles))
    counters["parent_count"] = len(canonical_smiles)
    return canonical_smiles, dict(counters)


def _score_and_filter_strict_flip(
    candidates: list[ClearCandidate],
    *,
    teacher: TeacherSemanticScorer,
    target_label: int,
) -> tuple[list[ClearCandidate], dict[str, int]]:
    if not teacher.available:
        raise RuntimeError(f"RF teacher is unavailable: {teacher.availability_reason}")
    retained: list[ClearCandidate] = []
    counters: Counter[str] = Counter()
    for index, candidate in enumerate(candidates, start=1):
        result = teacher.score_smiles(candidate.canonical_smiles, label=int(target_label))
        if not result.get("teacher_result_ok"):
            counters["rf_teacher_failed"] += 1
            continue
        probability_target = _as_float(result.get("teacher_prob"))
        predicted_label = _as_int(result.get("teacher_label"))
        if probability_target is None or predicted_label is None:
            counters["rf_probability_missing"] += 1
            continue
        candidate.rf_pred_label = predicted_label
        if int(target_label) == 1:
            candidate.rf_prob_label1 = float(probability_target)
            candidate.rf_prob_label0 = float(1.0 - probability_target)
        else:
            candidate.rf_prob_label0 = float(probability_target)
            candidate.rf_prob_label1 = float(1.0 - probability_target)
        candidate.rf_strict_flip = int(predicted_label) != int(target_label)
        candidate.rf_counterfactual_confidence = (
            candidate.rf_prob_label0 if int(target_label) == 1 else candidate.rf_prob_label1
        )
        if not candidate.rf_strict_flip:
            counters["rf_not_strict_flip"] += 1
            continue
        retained.append(candidate)
        if index % 500 == 0:
            print(
                f"[CLEAR_SELECTOR_RF_PROGRESS] scored={index}/{len(candidates)} retained={len(retained)}",
                flush=True,
            )
    counters["rf_strict_flip_candidate_count"] = len(retained)
    return retained, dict(counters)


def _sequence_digest(values: list[str], *, radius: int, bits: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"radius={radius};bits={bits}\n".encode("utf-8"))
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_cache(
    smiles_values: list[str],
    *,
    radius: int,
    bits: int,
    cache_path: Path,
) -> tuple[list[Any], bool, str]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError("RDKit is required for CLEAR Morgan selection.") from exc
    cache_key = _sequence_digest(smiles_values, radius=radius, bits=bits)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                stored_key = str(payload["cache_key"].item())
                bitstrings = [str(value) for value in payload["bitstrings"].tolist()]
            if stored_key == cache_key and len(bitstrings) == len(smiles_values):
                return [DataStructs.CreateFromBitString(value) for value in bitstrings], True, cache_key
        except Exception:
            pass
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(bits))
    fingerprints: list[Any] = []
    bitstrings: list[str] = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Canonical SMILES unexpectedly failed during fingerprinting: {smiles}")
        fingerprint = generator.GetFingerprint(mol)
        fingerprints.append(fingerprint)
        bitstrings.append(fingerprint.ToBitString())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_key=np.asarray(cache_key),
        bitstrings=np.asarray(bitstrings, dtype=f"<U{max(1, bits)}"),
    )
    return fingerprints, False, cache_key


def _coverage_cache(
    *,
    candidates: list[ClearCandidate],
    candidate_fps: list[Any],
    parent_fps: list[Any],
    candidate_key: str,
    parent_key: str,
    threshold: float,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    try:
        from rdkit import DataStructs
    except ImportError as exc:
        raise RuntimeError("RDKit is required for CLEAR Morgan coverage.") from exc
    cache_key = hashlib.sha256(
        f"{candidate_key}|{parent_key}|threshold={threshold:.17g}|bitorder=little".encode("utf-8")
    ).hexdigest()
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                stored_key = str(payload["cache_key"].item())
                packed = np.asarray(payload["coverage_bits"], dtype=np.uint8)
                counts = np.asarray(payload["coverage_counts"], dtype=np.int64)
            expected_width = (len(parent_fps) + 7) // 8
            if stored_key == cache_key and packed.shape == (len(candidates), expected_width):
                return packed, counts, True, cache_key
        except Exception:
            pass
    packed_rows: list[np.ndarray] = []
    counts: list[int] = []
    for index, fingerprint in enumerate(candidate_fps, start=1):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, parent_fps), dtype=np.float32
        )
        covered = similarities >= float(threshold)
        packed_rows.append(np.packbits(covered, bitorder="little"))
        counts.append(int(np.count_nonzero(covered)))
        if index % 500 == 0:
            print(f"[CLEAR_SELECTOR_COVERAGE_PROGRESS] candidates={index}/{len(candidates)}", flush=True)
    packed = np.stack(packed_rows, axis=0) if packed_rows else np.empty((0, 0), dtype=np.uint8)
    count_array = np.asarray(counts, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_key=np.asarray(cache_key),
        coverage_bits=packed,
        coverage_counts=count_array,
        num_parents=np.asarray(len(parent_fps), dtype=np.int64),
        threshold=np.asarray(float(threshold), dtype=np.float64),
    )
    return packed, count_array, False, cache_key


def _count_bits(packed: np.ndarray) -> int:
    return int(_POPCOUNT[np.asarray(packed, dtype=np.uint8)].sum(dtype=np.int64))


def _normalize_costs(candidates: list[ClearCandidate]) -> dict[str, Any]:
    finite = [float(candidate.total_action_cost) for candidate in candidates if candidate.total_action_cost is not None]
    minimum = min(finite) if finite else None
    maximum = max(finite) if finite else None
    missing = 0
    for candidate in candidates:
        if candidate.total_action_cost is None:
            candidate.normalized_cost = 1.0
            missing += 1
        elif minimum is None or maximum is None or math.isclose(minimum, maximum):
            candidate.normalized_cost = 0.0
        else:
            candidate.normalized_cost = float((candidate.total_action_cost - minimum) / (maximum - minimum))
    return {"cost_min": minimum, "cost_max": maximum, "missing_cost_count": missing}


def _select_topk(
    candidates: list[ClearCandidate],
    *,
    top_k: int,
    num_parents: int,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    # This pre-order mirrors the existing fragment selector's deterministic
    # first-in-list tie handling: CF utility, flip rate (constant here), then support.
    candidates.sort(
        key=lambda item: (
            -float(item.rf_counterfactual_confidence),
            -int(item.coverage_count),
            float(item.normalized_cost),
            item.canonical_smiles,
            int(item.source_order),
        )
    )
    selected_candidates: list[ClearCandidate] = []
    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    packed_width = len(candidates[0].coverage_bits) if candidates and candidates[0].coverage_bits is not None else 0
    covered = np.zeros(packed_width, dtype=np.uint8)
    for rank in range(1, top_k + 1):
        best: tuple[ClearCandidate, np.ndarray, int, float, float] | None = None
        best_score: float | None = None
        for candidate in candidates:
            if candidate.canonical_smiles in selected_ids or candidate.coverage_bits is None:
                continue
            new_bits = np.bitwise_and(candidate.coverage_bits, np.bitwise_not(covered))
            new_count = _count_bits(new_bits)
            coverage_gain = float(new_count / num_parents) if num_parents else 0.0
            max_similarity = 0.0
            if selected_candidates:
                from rdkit import DataStructs

                max_similarity = max(
                    float(value)
                    for value in DataStructs.BulkTanimotoSimilarity(
                        candidate.fingerprint,
                        [selected.fingerprint for selected in selected_candidates],
                    )
                )
            score = compute_weighted_mmr_score(
                cf_score=float(candidate.rf_counterfactual_confidence),
                coverage_gain=coverage_gain,
                cost_penalty=float(candidate.normalized_cost),
                max_similarity=max_similarity,
                w_cf=weights["w_cf"],
                w_cov=weights["w_cov"],
                w_cost=weights["w_cost"],
                w_red=weights["w_red"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best = (candidate, new_bits, new_count, coverage_gain, max_similarity)
        if best is None:
            break
        candidate, _new_bits, new_count, coverage_gain, max_similarity = best
        covered = np.bitwise_or(covered, candidate.coverage_bits)
        selected_ids.add(candidate.canonical_smiles)
        selected_candidates.append(candidate)
        selected_rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "candidate_smiles": candidate.canonical_smiles,
                "canonical_smiles": candidate.canonical_smiles,
                "source_method": candidate.source_method,
                "source_frequency": candidate.source_frequency,
                "rf_pred_label": candidate.rf_pred_label,
                "rf_prob_label0": candidate.rf_prob_label0,
                "rf_prob_label1": candidate.rf_prob_label1,
                "rf_strict_flip": candidate.rf_strict_flip,
                "official_target_success": candidate.official_target_success,
                "total_action_cost": candidate.total_action_cost,
                "normalized_cost": candidate.normalized_cost,
                "coverage_count": candidate.coverage_count,
                "new_coverage_count": new_count,
                "marginal_gain": best_score,
                "redundancy_penalty": max_similarity,
                "source_exp_id": candidate.source_exp_id,
                "source_instance_index": candidate.source_instance_index,
                "coverage_gain": coverage_gain,
                "rf_counterfactual_confidence": candidate.rf_counterfactual_confidence,
                "weighted_cf_component": weights["w_cf"] * candidate.rf_counterfactual_confidence,
                "weighted_coverage_component": weights["w_cov"] * coverage_gain,
                "weighted_cost_penalty": weights["w_cost"] * candidate.normalized_cost,
                "weighted_redundancy_penalty": weights["w_red"] * max_similarity,
                "cumulative_coverage_count": _count_bits(covered),
                "cumulative_coverage_rate": float(_count_bits(covered) / num_parents) if num_parents else 0.0,
            }
        )
    return selected_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--candidate-csv",
        default="outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv",
    )
    parser.add_argument(
        "--parent-csv",
        default=(
            "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/"
            "sft_v3_hiv_ppo_prompts_train_label1.csv"
        ),
    )
    parser.add_argument("--teacher-path", default="outputs/hpc/oracle/aids_rf_model.pkl")
    parser.add_argument("--out-dir", default="outputs/hpc/baselines/clear/aids/selected")
    parser.add_argument("--candidate-smiles-col", default="candidate_smiles")
    parser.add_argument("--parent-smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--coverage-metric", choices=("morgan_tanimoto",), default="morgan_tanimoto")
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--cache-dir", default="outputs/hpc/cache/clear_selector_morgan")
    parser.add_argument("--reference-selector-summary", default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--w-cov", type=float, default=None)
    parser.add_argument("--w-cf", type=float, default=None)
    parser.add_argument("--w-cost", type=float, default=None)
    parser.add_argument("--w-red", type=float, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("[ERROR] --top-k must be positive.")
    if int(args.target_label) not in (0, 1):
        raise SystemExit("[ERROR] --target-label must be 0 or 1.")
    np.random.seed(int(args.seed))
    if not 0.0 <= float(args.coverage_threshold) <= 1.0:
        raise SystemExit("[ERROR] --coverage-threshold must be within [0, 1].")
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    parent_csv = Path(args.parent_csv).expanduser().resolve()
    teacher_path = Path(args.teacher_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    selected_csv = out_dir / "clear_aids_rf_strict_flip_top20_greedy_mmr.csv"
    summary_json = out_dir / "clear_aids_rf_strict_flip_top20_greedy_mmr_summary.json"
    trace_csv = out_dir / "clear_aids_rf_strict_flip_top20_greedy_mmr_trace.csv"
    for name, path in (("candidate CSV", candidate_csv), ("parent CSV", parent_csv), ("teacher", teacher_path)):
        if not path.is_file():
            raise SystemExit(f"[ERROR] {name} does not exist: {path}")

    weights, weights_source = _load_weights(args)
    print("[CLEAR_GLOBAL_SELECTOR_CONFIG]", flush=True)
    print(f"candidate_csv={candidate_csv}", flush=True)
    print(f"parent_csv={parent_csv}", flush=True)
    print(f"teacher_path={teacher_path}", flush=True)
    print(f"coverage_threshold={args.coverage_threshold}", flush=True)
    print(f"weights={weights}", flush=True)
    print(f"weights_source={weights_source}", flush=True)

    candidates, candidate_stats = _load_and_deduplicate_candidates(
        candidate_csv, smiles_col=args.candidate_smiles_col
    )
    parents, parent_stats = _load_parents(
        parent_csv,
        smiles_col=args.parent_smiles_col,
        label_col=args.label_col,
        target_label=int(args.target_label),
    )
    teacher = TeacherSemanticScorer(teacher_path)
    candidates, rf_stats = _score_and_filter_strict_flip(
        candidates, teacher=teacher, target_label=int(args.target_label)
    )
    failure_summary = {
        "selector_name": SELECTOR_NAME,
        "selector_version": SELECTOR_VERSION,
        "top_k": int(args.top_k),
        **weights,
        "weights_source": weights_source,
        "coverage_metric": args.coverage_metric,
        "coverage_threshold": float(args.coverage_threshold),
        "candidate_count_before_filter": int(candidate_stats.get("candidate_count_before_filter", 0)),
        "unique_candidate_count": int(candidate_stats.get("unique_candidate_count", 0)),
        "rf_strict_flip_candidate_count": len(candidates),
        "selected_count": 0,
        "selection_ok": False,
        "candidate_filter_stats": candidate_stats,
        "rf_filter_stats": rf_stats,
        "parent_stats": parent_stats,
        "candidate_csv": str(candidate_csv),
        "parent_csv": str(parent_csv),
        "teacher_path": str(teacher_path),
        "CF_MODE": "strict_flip",
        "random_seed": int(args.seed),
    }
    if len(candidates) < int(args.top_k):
        failure_summary["failure_reason"] = "strict_flip_unique_candidates_below_top_k"
        _write_csv(selected_csv, [], OUTPUT_FIELDS)
        _write_csv(trace_csv, [], TRACE_FIELDS)
        _write_json(summary_json, failure_summary)
        print(
            f"[CLEAR_GLOBAL_SELECTOR_FAILED] strict_flip_candidates={len(candidates)} top_k={args.top_k}",
            flush=True,
        )
        return 2
    if not parents:
        failure_summary["failure_reason"] = "no_valid_target_label_parents"
        _write_csv(selected_csv, [], OUTPUT_FIELDS)
        _write_csv(trace_csv, [], TRACE_FIELDS)
        _write_json(summary_json, failure_summary)
        print("[CLEAR_GLOBAL_SELECTOR_FAILED] no valid parents", flush=True)
        return 2

    cost_stats = _normalize_costs(candidates)
    parent_fps, parent_fp_hit, parent_key = _fingerprint_cache(
        parents,
        radius=int(args.fingerprint_radius),
        bits=int(args.fingerprint_bits),
        cache_path=cache_dir / "parent_fingerprints.npz",
    )
    candidate_smiles = [candidate.canonical_smiles for candidate in candidates]
    candidate_fps, candidate_fp_hit, candidate_key = _fingerprint_cache(
        candidate_smiles,
        radius=int(args.fingerprint_radius),
        bits=int(args.fingerprint_bits),
        cache_path=cache_dir / "candidate_fingerprints.npz",
    )
    coverage_bits, coverage_counts, coverage_hit, coverage_key = _coverage_cache(
        candidates=candidates,
        candidate_fps=candidate_fps,
        parent_fps=parent_fps,
        candidate_key=candidate_key,
        parent_key=parent_key,
        threshold=float(args.coverage_threshold),
        cache_path=cache_dir / "coverage_bitsets.npz",
    )
    for index, candidate in enumerate(candidates):
        candidate.fingerprint = candidate_fps[index]
        candidate.coverage_bits = coverage_bits[index]
        candidate.coverage_count = int(coverage_counts[index])

    selected_rows = _select_topk(
        candidates,
        top_k=int(args.top_k),
        num_parents=len(parents),
        weights=weights,
    )
    unique_selected = {str(row["canonical_smiles"]) for row in selected_rows}
    selection_ok = bool(
        len(selected_rows) == int(args.top_k)
        and len(unique_selected) == int(args.top_k)
        and all(bool(row["rf_strict_flip"]) for row in selected_rows)
        and [int(row["rank"]) for row in selected_rows] == list(range(1, int(args.top_k) + 1))
    )
    _write_csv(selected_csv, selected_rows, OUTPUT_FIELDS)
    _write_csv(trace_csv, selected_rows, TRACE_FIELDS)
    summary = {
        **failure_summary,
        "selected_count": len(selected_rows),
        "selection_ok": selection_ok,
        "cost_stats": cost_stats,
        "fingerprint_radius": int(args.fingerprint_radius),
        "fingerprint_bits": int(args.fingerprint_bits),
        "cache_dir": str(cache_dir),
        "parent_fingerprint_cache_hit": parent_fp_hit,
        "candidate_fingerprint_cache_hit": candidate_fp_hit,
        "coverage_cache_hit": coverage_hit,
        "coverage_cache_key": coverage_key,
        "coverage_storage": "numpy_packbits_little_endian",
        "full_float_parent_candidate_matrix_saved": False,
        "final_cumulative_coverage_count": (
            int(selected_rows[-1]["cumulative_coverage_count"]) if selected_rows else 0
        ),
        "final_cumulative_coverage_rate": (
            float(selected_rows[-1]["cumulative_coverage_rate"]) if selected_rows else 0.0
        ),
        "selected_csv": str(selected_csv),
        "trace_csv": str(trace_csv),
        "selection_formula": (
            "w_cov*new_coverage + w_cf*rf_counterfactual_confidence "
            "- w_cost*normalized_total_action_cost - w_red*max_morgan_similarity"
        ),
        "tie_break": "existing_selector_first_in_preordered_list",
        "node_fgw_used_for_selection": False,
    }
    _write_json(summary_json, summary)
    if not selection_ok:
        print("[CLEAR_GLOBAL_SELECTOR_FAILED] output invariant check failed", flush=True)
        return 2
    print("[CLEAR_GLOBAL_SELECTOR_DONE]", flush=True)
    print(f"selected_csv={selected_csv}", flush=True)
    print(f"selected_count={len(selected_rows)}", flush=True)
    print(f"final_cumulative_coverage_rate={summary['final_cumulative_coverage_rate']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
