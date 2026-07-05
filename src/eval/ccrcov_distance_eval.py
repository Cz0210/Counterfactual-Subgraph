"""CCRCov evaluation using learned GREED-GED or MolCLR embedding distances."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.eval.close_counterfactual_coverage import (
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    CandidateRecord,
    ParentRecord,
    hard_delete_substructure_any_match,
    predict_with_teacher,
    render_report,
    _as_bool,
    _as_float,
    _load_candidate_records,
    _load_parent_records,
    _row_base,
)
from src.eval.greed_distance.infer import GreedDistancePredictor
from src.eval.greed_distance.pair_generation import GT_FULLGRAPH_FIELDS, OURS_FRAGMENT_FIELDS
from src.eval.molclr_distance import MolCLREmbeddingDistanceLookup
from src.rewards.teacher_semantic import TeacherSemanticScorer
from src.utils.io import ensure_directory

try:  # pragma: no cover - optional plotting dependency
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


EXTRA_SUMMARY_FIELDS = [
    "cf_mode",
    "min_cf_drop",
    "SuppCov",
    "StructRed",
    "CovRed",
    "ValidRate",
    "avg_size_among_covered",
]
DISTANCE_SUMMARY_FIELDS: list[str] = []
for _field in SUMMARY_FIELDS:
    DISTANCE_SUMMARY_FIELDS.append(_field)
    if _field == "ged_mode":
        for _extra in ("cf_mode", "min_cf_drop"):
            if _extra not in DISTANCE_SUMMARY_FIELDS:
                DISTANCE_SUMMARY_FIELDS.append(_extra)
for _field in EXTRA_SUMMARY_FIELDS:
    if _field not in DISTANCE_SUMMARY_FIELDS:
        DISTANCE_SUMMARY_FIELDS.append(_field)

CF_MODES = ("strict_flip", "drop_or_flip", "drop_only")
GT_DIRECTORY_CANDIDATES = (
    "gt_selected_fullgraphs.csv",
    "selected_fullgraphs.csv",
    "selected_subgraphs.csv",
    "selected_subgraphs.json",
    "selected_subgraphs.jsonl",
    "candidate_pool.jsonl",
    "candidate_pool.csv",
)
OURS_DIRECTORY_CANDIDATES = (
    "selected_subgraphs.csv",
    "selected_subgraphs.json",
    "selected_subgraphs.jsonl",
    "candidate_pool.jsonl",
    "candidate_pool.csv",
)


class DistanceProvider(Protocol):
    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        ...


class GreedDistanceProvider:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cuda") -> None:
        self.predictor = GreedDistancePredictor(checkpoint_path, device=device)

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        result = self.predictor.predict_smiles(smiles_a, smiles_b)
        return {
            "distance": result.get("distance"),
            "cosine_similarity": None,
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
        }


class MolCLRDistanceProvider:
    def __init__(self, embedding_dir: str | Path) -> None:
        self.lookup = MolCLREmbeddingDistanceLookup(embedding_dir)

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        return self.lookup.distance(smiles_a, smiles_b)


def normalize_cf_mode(cf_mode: str | None) -> str:
    mode = str(cf_mode or "strict_flip").strip()
    if mode not in CF_MODES:
        raise ValueError(f"Unsupported cf_mode={mode!r}. Expected one of: {', '.join(CF_MODES)}")
    return mode


def _parse_int_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Sequence[float | None] | Any) -> float | None:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return float(sum(clean) / len(clean)) if clean else None


def _median(values: Sequence[float | None] | Any) -> float | None:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return None
    clean.sort()
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return float(clean[midpoint])
    return float((clean[midpoint - 1] + clean[midpoint]) / 2.0)


def _threshold_similarity_equivalent(distance_type: str, threshold: float) -> float | None:
    if distance_type != "embedding":
        return None
    return max(-1.0, min(1.0, float(1.0 - threshold)))


def _strict_flip(row: dict[str, Any], *, label: int) -> bool:
    pred_after = _parse_int_label(row.get("pred_after"))
    return bool(pred_after is not None and pred_after != int(label))


def _row_flip_value(row: dict[str, Any]) -> float:
    label = _parse_int_label(row.get("label"))
    if label is None:
        return 1.0 if _as_bool(row.get("cf_flip")) else 0.0
    return 1.0 if _strict_flip(row, label=label) else 0.0


def _cf_condition_for_mode(
    row: dict[str, Any],
    *,
    label: int,
    cf_mode: str,
    min_cf_drop: float,
) -> bool:
    """Evaluate the counterfactual condition for an already-close action row."""

    mode = normalize_cf_mode(cf_mode)
    cf_drop = _as_float(row.get("cf_drop"))
    cf_flip = _strict_flip(row, label=label)
    drop_ok = cf_drop is not None and cf_drop >= float(min_cf_drop)
    if mode == "strict_flip":
        return bool(cf_flip)
    if mode == "drop_or_flip":
        return bool(cf_flip or drop_ok)
    if mode == "drop_only":
        return bool(drop_ok)
    raise ValueError(f"Unsupported cf_mode={mode!r}")


def _pick_best_row_for_mode(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    label: int,
    cf_mode: str,
    min_cf_drop: float,
) -> tuple[dict[str, Any] | None, bool, bool]:
    eligible: list[tuple[bool, float, float, dict[str, Any]]] = []
    close_only = False
    close_cf = False
    for row in rows:
        distance = _as_float(row.get("distance"))
        if distance is None or distance > float(threshold):
            continue
        close_only = True
        is_cf = _cf_condition_for_mode(row, label=label, cf_mode=cf_mode, min_cf_drop=min_cf_drop)
        close_cf = close_cf or is_cf
        cf_drop = _as_float(row.get("cf_drop"))
        eligible.append((is_cf, distance, -(cf_drop if cf_drop is not None else -1e9), row))
    if not eligible:
        return None, close_only, close_cf
    eligible.sort(key=lambda item: (not item[0], item[1], item[2]))
    return eligible[0][3], close_only, close_cf


def build_threshold_summary_for_cf_mode(
    detail_rows: list[dict[str, Any]],
    *,
    method: str,
    distance_type: str,
    ged_mode: str,
    thresholds: Sequence[float],
    total_parents: int,
    total_candidates: int,
    cf_mode: str,
    min_cf_drop: float,
) -> list[dict[str, Any]]:
    mode = normalize_cf_mode(cf_mode)
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
            best_row, close_only, close_cf = _pick_best_row_for_mode(
                rows,
                threshold=float(threshold),
                label=label,
                cf_mode=mode,
                min_cf_drop=float(min_cf_drop),
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
                "cf_mode": mode,
                "min_cf_drop": float(min_cf_drop),
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
                "flip_rate_among_covered": _mean(_row_flip_value(row) for row in best_rows),
                "avg_atom_delete_ratio_among_covered": _mean(
                    _as_float(row.get("atom_delete_ratio")) for row in best_rows
                ),
                "avg_bond_delete_ratio_among_covered": _mean(
                    _as_float(row.get("bond_delete_ratio")) for row in best_rows
                ),
                "cache_hit_rate": 0.0,
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
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict, tuple))
                    else ("" if row.get(key) is None else row.get(key))
                    for key in fieldnames
                }
            )


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser().resolve()
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fingerprint(smiles: str) -> Any | None:
    if Chem is None or DataStructs is None or rdFingerprintGenerator is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(mol)


def _structural_redundancy(candidates: list[CandidateRecord]) -> float | None:
    if DataStructs is None:
        return None
    fps = [fp for fp in (_fingerprint(candidate.smiles) for candidate in candidates) if fp is not None]
    if len(fps) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, fp_i in enumerate(fps):
        for fp_j in fps[i + 1 :]:
            total += float(DataStructs.TanimotoSimilarity(fp_i, fp_j))
            count += 1
    return float(total / count) if count else 0.0


def _coverage_redundancy(
    detail_rows: list[dict[str, Any]],
    *,
    threshold: float,
    cf_mode: str,
    min_cf_drop: float,
) -> float | None:
    cover_sets: dict[str, set[str]] = {}
    for row in detail_rows:
        distance = _as_float(row.get("distance"))
        if distance is None or distance > float(threshold):
            continue
        label = int(float(row.get("label") or 0))
        if not _cf_condition_for_mode(row, label=label, cf_mode=cf_mode, min_cf_drop=min_cf_drop):
            continue
        cover_sets.setdefault(str(row.get("candidate_id")), set()).add(str(row.get("parent_id")))
    sets = [value for value in cover_sets.values() if value]
    if len(sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            union = len(left | right)
            total += (len(left & right) / union) if union else 0.0
            count += 1
    return float(total / count) if count else 0.0


def _augment_summary(
    summaries: list[dict[str, Any]],
    *,
    detail_rows: list[dict[str, Any]],
    candidates: list[CandidateRecord],
    cf_mode: str,
    min_cf_drop: float,
) -> list[dict[str, Any]]:
    struct_red = _structural_redundancy(candidates)
    total_rows = len(detail_rows)
    valid_rate = (
        sum(1 for row in detail_rows if _as_float(row.get("distance")) is not None and row.get("p_after") not in {None, ""})
        / total_rows
        if total_rows
        else 0.0
    )
    for row in summaries:
        threshold = float(row.get("threshold") or 0.0)
        row["SuppCov"] = row.get("match_rate") if str(row.get("method", "")).startswith("ours") else None
        row["StructRed"] = struct_red
        row["CovRed"] = _coverage_redundancy(
            detail_rows,
            threshold=threshold,
            cf_mode=cf_mode,
            min_cf_drop=min_cf_drop,
        )
        row["ValidRate"] = valid_rate
        row["avg_size_among_covered"] = row.get("avg_atom_delete_ratio_among_covered")
    return summaries


def _save_figures(summary_rows: list[dict[str, Any]], figures_dir: Path, title_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ensure_directory(figures_dir)

    def plot(metric: str, filename: str, ylabel: str, percent: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        by_method: dict[str, list[dict[str, Any]]] = {}
        for row in summary_rows:
            by_method.setdefault(str(row.get("method")), []).append(row)
        for method, rows in by_method.items():
            rows = sorted(rows, key=lambda item: float(item.get("threshold") or 0.0))
            xs = [float(row.get("threshold") or 0.0) for row in rows]
            ys = []
            for row in rows:
                value = _as_float(row.get(metric))
                ys.append((value or 0.0) * 100.0 if percent else (value or 0.0))
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=method)
        ax.set_title(f"{title_prefix} {metric}".strip())
        ax.set_xlabel("threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        for suffix in (".png", ".pdf"):
            out = figures_dir / f"{filename}{suffix}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[PLOT_SAVED] {out}", flush=True)
        plt.close(fig)

    plot("close_cf_coverage", "coverage_vs_threshold_close_cf", "coverage (%)", percent=True)
    plot("close_only_coverage", "coverage_vs_threshold_close_only", "coverage (%)", percent=True)
    plot("avg_best_distance", "cost_vs_threshold", "avg best distance")
    plot("avg_cf_drop_among_covered", "cfdrop_vs_threshold", "mean CFDrop")
    plot("flip_rate_among_covered", "fliprate_vs_threshold", "flip rate", percent=True)


def _write_method_outputs(
    *,
    output_dir: Path,
    method: str,
    distance_type: str,
    distance_name: str,
    thresholds: Sequence[float],
    details: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    cf_mode: str,
    min_cf_drop: float,
) -> dict[str, str]:
    ensure_directory(output_dir)
    _write_csv(output_dir / "details.csv", details, DETAIL_FIELDS)
    _write_csv(output_dir / "threshold_summary.csv", summaries, DISTANCE_SUMMARY_FIELDS)
    _write_json(output_dir / "threshold_summary.json", {"threshold_summary": summaries})
    report = render_report(
        method_name=method,
        distance_type=distance_type,
        ged_mode=distance_name,
        thresholds=thresholds,
        summaries=summaries,
        total_parents=total_parents,
        total_candidates=total_candidates,
        match_metric=method.startswith("ours"),
    )
    report += f"\nCF mode: {cf_mode}\n"
    report += f"Min CFDrop: {float(min_cf_drop):.6g}\n"
    report += "\nAdditional metrics: SuppCov, StructRed, CovRed, ValidRate are included in threshold_summary.csv.\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_figures(summaries, output_dir / "figures", f"{method} {distance_name}")
    return {
        "details_csv": str(output_dir / "details.csv"),
        "threshold_summary_csv": str(output_dir / "threshold_summary.csv"),
        "report_md": str(output_dir / "report.md"),
    }


def _distance_row_update(
    row: dict[str, Any],
    *,
    provider: DistanceProvider,
    parent_smiles: str,
    action_smiles: str,
    distance_type: str,
) -> None:
    result = provider.distance(parent_smiles, action_smiles)
    row.update(
        {
            "distance": result.get("distance"),
            "cosine_similarity": result.get("cosine_similarity"),
            "embedding_ok": bool(result.get("ok")) if distance_type == "embedding" else None,
            "ged_ok": bool(result.get("ok")) if distance_type == "ged" else None,
            "error": row.get("error") or result.get("error"),
        }
    )


def _evaluate_ours(
    *,
    parents: list[ParentRecord],
    candidates: list[CandidateRecord],
    teacher: Any,
    provider: DistanceProvider,
    distance_type: str,
    distance_name: str,
    partial_path: Path,
    partial_every: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    before_cache: dict[str, dict[str, Any]] = {}
    total_work = max(1, len(parents) * max(1, len(candidates)))
    work_done = 0
    for parent in parents:
        before = before_cache.setdefault(parent.smiles, predict_with_teacher(teacher, parent.smiles, parent.label))
        for candidate in candidates:
            work_done += 1
            base = _row_base(method="ours_selected_subgraphs", distance_type=distance_type, ged_mode=distance_name, parent=parent, candidate=candidate)
            deletions = hard_delete_substructure_any_match(parent.smiles, candidate.smiles)
            if not deletions:
                row = dict(base)
                row.update({"p_before": before.get("p_label"), "pred_before": before.get("pred_label"), "error": "no_substructure_match"})
                details.append(row)
                continue
            for deletion in deletions:
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
                residual = str(deletion.get("residual_smiles") or "")
                if deletion.get("delete_valid") and residual:
                    after = predict_with_teacher(teacher, residual, parent.label)
                    row.update(
                        {
                            "p_after": after.get("p_label"),
                            "pred_after": after.get("pred_label"),
                            "cf_drop": (before.get("p_label") - after.get("p_label"))
                            if before.get("p_label") is not None and after.get("p_label") is not None
                            else None,
                            "cf_flip": after.get("pred_label") is not None and int(after.get("pred_label")) != int(parent.label),
                        }
                    )
                    _distance_row_update(row, provider=provider, parent_smiles=parent.smiles, action_smiles=residual, distance_type=distance_type)
                details.append(row)
            if partial_every > 0 and work_done % int(partial_every) == 0:
                _write_csv(partial_path, details, DETAIL_FIELDS)
                print(f"[PROGRESS] ours pairs={work_done}/{total_work} detail_rows={len(details)}", flush=True)
    _write_csv(partial_path, details, DETAIL_FIELDS)
    return details


def _evaluate_gt_fullgraph(
    *,
    parents: list[ParentRecord],
    candidates: list[CandidateRecord],
    teacher: Any,
    provider: DistanceProvider,
    distance_type: str,
    distance_name: str,
    method: str,
    partial_path: Path,
    partial_every: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    before_cache: dict[str, dict[str, Any]] = {}
    total_work = max(1, len(parents) * max(1, len(candidates)))
    work_done = 0
    for parent in parents:
        before = before_cache.setdefault(parent.smiles, predict_with_teacher(teacher, parent.smiles, parent.label))
        for candidate in candidates:
            work_done += 1
            row = _row_base(method=method, distance_type=distance_type, ged_mode=distance_name, parent=parent, candidate=candidate)
            row.update(
                {
                    "fragment_smiles": "",
                    "match": True,
                    "delete_valid": True,
                    "residual_smiles": candidate.smiles,
                    "p_before": before.get("p_label"),
                    "pred_before": before.get("pred_label"),
                }
            )
            after = predict_with_teacher(teacher, candidate.smiles, parent.label)
            row.update(
                {
                    "p_after": after.get("p_label"),
                    "pred_after": after.get("pred_label"),
                    "cf_drop": (before.get("p_label") - after.get("p_label"))
                    if before.get("p_label") is not None and after.get("p_label") is not None
                    else None,
                    "cf_flip": after.get("pred_label") is not None and int(after.get("pred_label")) != int(parent.label),
                    "error": after.get("error"),
                }
            )
            _distance_row_update(row, provider=provider, parent_smiles=parent.smiles, action_smiles=candidate.smiles, distance_type=distance_type)
            details.append(row)
            if partial_every > 0 and work_done % int(partial_every) == 0:
                _write_csv(partial_path, details, DETAIL_FIELDS)
                print(f"[PROGRESS] {method} pairs={work_done}/{total_work} detail_rows={len(details)}", flush=True)
    _write_csv(partial_path, details, DETAIL_FIELDS)
    return details


def _evaluate_fullgraph_method(
    *,
    output: Path,
    output_key: str,
    output_dir_name: str,
    method: str,
    parents: list[ParentRecord],
    candidates: list[CandidateRecord],
    teacher: Any,
    provider: DistanceProvider,
    distance_type: str,
    distance_name: str,
    thresholds: Sequence[float],
    cf_mode: str,
    min_cf_drop: float,
    partial_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_dir = ensure_directory(output / output_dir_name)
    details = _evaluate_gt_fullgraph(
        parents=parents,
        candidates=candidates,
        teacher=teacher,
        provider=provider,
        distance_type=distance_type,
        distance_name=distance_name,
        method=method,
        partial_path=method_dir / "details.partial.csv",
        partial_every=int(partial_every),
    )
    summaries = build_threshold_summary_for_cf_mode(
        details,
        method=method,
        distance_type=distance_type,
        ged_mode=distance_name,
        thresholds=thresholds,
        total_parents=len(parents),
        total_candidates=len(candidates),
        cf_mode=cf_mode,
        min_cf_drop=float(min_cf_drop),
    )
    summaries = _augment_summary(
        summaries,
        detail_rows=details,
        candidates=candidates,
        cf_mode=cf_mode,
        min_cf_drop=float(min_cf_drop),
    )
    output_paths = _write_method_outputs(
        output_dir=method_dir,
        method=method,
        distance_type=distance_type,
        distance_name=distance_name,
        thresholds=thresholds,
        details=details,
        summaries=summaries,
        total_parents=len(parents),
        total_candidates=len(candidates),
        cf_mode=cf_mode,
        min_cf_drop=float(min_cf_drop),
    )
    return summaries, {output_key: output_paths}


def evaluate_ccrcov_with_distance(
    *,
    dataset_csv: str | Path,
    ours_selected_path: str | Path | None,
    gt_fullgraph_candidates_path: str | Path | None,
    teacher_path: str | Path,
    provider: DistanceProvider,
    distance_type: str,
    distance_name: str,
    label: int,
    thresholds: Sequence[float],
    output_root: str | Path,
    clear_fullgraph_candidates_path: str | Path | None = None,
    smiles_col: str = "smiles",
    label_col: str = "label",
    max_parents: int | None = None,
    max_candidates: int | None = None,
    require_flip_only: bool = False,
    cf_mode: str = "strict_flip",
    min_cf_drop: float = 0.0,
    partial_every: int = 5000,
) -> dict[str, Any]:
    started = time.time()
    mode = "strict_flip" if require_flip_only else normalize_cf_mode(cf_mode)
    output = ensure_directory(Path(output_root).expanduser().resolve())
    print(f"[TASK_START] distance={distance_name} cf_mode={mode} output_root={output}", flush=True)
    _dataset_path, parents, _actual_label_col = _load_parent_records(
        dataset_csv,
        label=int(label),
        smiles_col=smiles_col,
        label_col=label_col,
        max_parents=max_parents,
    )
    ours_candidates: list[CandidateRecord] = []
    gt_candidates: list[CandidateRecord] = []
    clear_candidates: list[CandidateRecord] = []
    if ours_selected_path:
        _ours_path, ours_candidates = _load_candidate_records(
            ours_selected_path,
            fields=OURS_FRAGMENT_FIELDS,
            directory_candidates=OURS_DIRECTORY_CANDIDATES,
        )
    if gt_fullgraph_candidates_path:
        _gt_path, gt_candidates = _load_candidate_records(
            gt_fullgraph_candidates_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=GT_DIRECTORY_CANDIDATES,
        )
    if clear_fullgraph_candidates_path:
        _clear_path, clear_candidates = _load_candidate_records(
            clear_fullgraph_candidates_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=GT_DIRECTORY_CANDIDATES,
        )
    if max_candidates is not None:
        ours_candidates = ours_candidates[: int(max_candidates)]
        gt_candidates = gt_candidates[: int(max_candidates)]
        clear_candidates = clear_candidates[: int(max_candidates)]
    print(
        f"[PROGRESS] parents={len(parents)} ours_candidates={len(ours_candidates)} "
        f"gt_fullgraph_candidates={len(gt_candidates)} clear_rf_fullgraph_candidates={len(clear_candidates)}",
        flush=True,
    )
    if not ours_candidates and not gt_candidates and not clear_candidates:
        raise ValueError("No candidates loaded for any method.")
    teacher = TeacherSemanticScorer(teacher_path)

    all_summaries: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    if ours_candidates:
        ours_dir = ensure_directory(output / f"ours_{distance_name}")
        ours_details = _evaluate_ours(
            parents=parents,
            candidates=ours_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=distance_type,
            distance_name=distance_name,
            partial_path=ours_dir / "details.partial.csv",
            partial_every=int(partial_every),
        )
        ours_summaries = build_threshold_summary_for_cf_mode(
            ours_details,
            method="ours_selected_subgraphs",
            distance_type=distance_type,
            ged_mode=distance_name,
            thresholds=thresholds,
            total_parents=len(parents),
            total_candidates=len(ours_candidates),
            cf_mode=mode,
            min_cf_drop=float(min_cf_drop),
        )
        ours_summaries = _augment_summary(
            ours_summaries,
            detail_rows=ours_details,
            candidates=ours_candidates,
            cf_mode=mode,
            min_cf_drop=float(min_cf_drop),
        )
        outputs["ours"] = _write_method_outputs(
            output_dir=ours_dir,
            method="ours_selected_subgraphs",
            distance_type=distance_type,
            distance_name=distance_name,
            thresholds=thresholds,
            details=ours_details,
            summaries=ours_summaries,
            total_parents=len(parents),
            total_candidates=len(ours_candidates),
            cf_mode=mode,
            min_cf_drop=float(min_cf_drop),
        )
        all_summaries.extend(ours_summaries)

    if gt_candidates:
        gt_summaries, gt_outputs = _evaluate_fullgraph_method(
            output=output,
            output_key="gt_fullgraph",
            output_dir_name=f"gt_fullgraph_{distance_name}",
            method="gt_fullgraph",
            parents=parents,
            candidates=gt_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=distance_type,
            distance_name=distance_name,
            thresholds=thresholds,
            cf_mode=mode,
            min_cf_drop=float(min_cf_drop),
            partial_every=int(partial_every),
        )
        outputs.update(gt_outputs)
        all_summaries.extend(gt_summaries)

    if clear_candidates:
        clear_summaries, clear_outputs = _evaluate_fullgraph_method(
            output=output,
            output_key="clear_rf_fullgraph",
            output_dir_name=f"clear_rf_fullgraph_{distance_name}",
            method="CLEAR-RF-FullGraph",
            parents=parents,
            candidates=clear_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=distance_type,
            distance_name=distance_name,
            thresholds=thresholds,
            cf_mode=mode,
            min_cf_drop=float(min_cf_drop),
            partial_every=int(partial_every),
        )
        outputs.update(clear_outputs)
        all_summaries.extend(clear_summaries)

    combined_dir = ensure_directory(output / "combined")
    _write_csv(combined_dir / "combined_threshold_summary.csv", all_summaries, DISTANCE_SUMMARY_FIELDS)
    _write_json(combined_dir / "combined_threshold_summary.json", {"threshold_summary": all_summaries})
    _save_figures(all_summaries, combined_dir / "figures", f"CCRCov {distance_name}")
    elapsed = time.time() - started
    (combined_dir / "combined_report.md").write_text(
        "# CCRCov Distance Evaluation\n\n"
        f"- distance_name: {distance_name}\n"
        f"- distance_type: {distance_type}\n"
        f"- CF mode: {mode}\n"
        f"- min_cf_drop: {float(min_cf_drop):.6g}\n"
        f"- elapsed_sec: {elapsed:.2f}\n"
        "- fullgraph baseline distance source: learned/embedding provider, not NetworkX GED by default.\n\n"
        "See `combined_threshold_summary.csv` and per-method reports for metrics.\n",
        encoding="utf-8",
    )
    print(f"[TASK_DONE] output_root={output} elapsed_sec={elapsed:.2f}", flush=True)
    return {"output_root": str(output), "outputs": outputs, "combined_summary": str(combined_dir / "combined_threshold_summary.csv")}
