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
    build_threshold_summary,
    hard_delete_substructure_any_match,
    predict_with_teacher,
    render_report,
    _as_bool,
    _as_float,
    _cf_condition,
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
    "SuppCov",
    "StructRed",
    "CovRed",
    "ValidRate",
    "avg_size_among_covered",
]
DISTANCE_SUMMARY_FIELDS = list(SUMMARY_FIELDS) + [field for field in EXTRA_SUMMARY_FIELDS if field not in SUMMARY_FIELDS]
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
    require_flip_only: bool,
    min_cf_drop: float,
) -> float | None:
    cover_sets: dict[str, set[str]] = {}
    for row in detail_rows:
        distance = _as_float(row.get("distance"))
        if distance is None or distance > float(threshold):
            continue
        label = int(float(row.get("label") or 0))
        if not _cf_condition(row, label=label, require_flip_only=require_flip_only, min_cf_drop=min_cf_drop):
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
    require_flip_only: bool,
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
            require_flip_only=require_flip_only,
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
            row = _row_base(method="gt_fullgraph", distance_type=distance_type, ged_mode=distance_name, parent=parent, candidate=candidate)
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
                print(f"[PROGRESS] gt_fullgraph pairs={work_done}/{total_work} detail_rows={len(details)}", flush=True)
    _write_csv(partial_path, details, DETAIL_FIELDS)
    return details


def evaluate_ccrcov_with_distance(
    *,
    dataset_csv: str | Path,
    ours_selected_path: str | Path,
    gt_fullgraph_candidates_path: str | Path,
    teacher_path: str | Path,
    provider: DistanceProvider,
    distance_type: str,
    distance_name: str,
    label: int,
    thresholds: Sequence[float],
    output_root: str | Path,
    smiles_col: str = "smiles",
    label_col: str = "label",
    max_parents: int | None = None,
    max_candidates: int | None = None,
    require_flip_only: bool = False,
    min_cf_drop: float = 0.0,
    partial_every: int = 5000,
) -> dict[str, Any]:
    started = time.time()
    output = ensure_directory(Path(output_root).expanduser().resolve())
    print(f"[TASK_START] distance={distance_name} output_root={output}", flush=True)
    _dataset_path, parents, _actual_label_col = _load_parent_records(
        dataset_csv,
        label=int(label),
        smiles_col=smiles_col,
        label_col=label_col,
        max_parents=max_parents,
    )
    _ours_path, ours_candidates = _load_candidate_records(
        ours_selected_path,
        fields=OURS_FRAGMENT_FIELDS,
        directory_candidates=OURS_DIRECTORY_CANDIDATES,
    )
    _gt_path, gt_candidates = _load_candidate_records(
        gt_fullgraph_candidates_path,
        fields=GT_FULLGRAPH_FIELDS,
        directory_candidates=GT_DIRECTORY_CANDIDATES,
    )
    if max_candidates is not None:
        ours_candidates = ours_candidates[: int(max_candidates)]
        gt_candidates = gt_candidates[: int(max_candidates)]
    print(
        f"[PROGRESS] parents={len(parents)} ours_candidates={len(ours_candidates)} gt_fullgraph_candidates={len(gt_candidates)}",
        flush=True,
    )
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
        ours_summaries = build_threshold_summary(
            ours_details,
            method="ours_selected_subgraphs",
            distance_type=distance_type,
            ged_mode=distance_name,
            thresholds=thresholds,
            total_parents=len(parents),
            total_candidates=len(ours_candidates),
            require_flip_only=require_flip_only,
            min_cf_drop=float(min_cf_drop),
        )
        ours_summaries = _augment_summary(
            ours_summaries,
            detail_rows=ours_details,
            candidates=ours_candidates,
            require_flip_only=require_flip_only,
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
        )
        all_summaries.extend(ours_summaries)

    if gt_candidates:
        gt_dir = ensure_directory(output / f"gt_fullgraph_{distance_name}")
        gt_details = _evaluate_gt_fullgraph(
            parents=parents,
            candidates=gt_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=distance_type,
            distance_name=distance_name,
            partial_path=gt_dir / "details.partial.csv",
            partial_every=int(partial_every),
        )
        gt_summaries = build_threshold_summary(
            gt_details,
            method="gt_fullgraph",
            distance_type=distance_type,
            ged_mode=distance_name,
            thresholds=thresholds,
            total_parents=len(parents),
            total_candidates=len(gt_candidates),
            require_flip_only=require_flip_only,
            min_cf_drop=float(min_cf_drop),
        )
        gt_summaries = _augment_summary(
            gt_summaries,
            detail_rows=gt_details,
            candidates=gt_candidates,
            require_flip_only=require_flip_only,
            min_cf_drop=float(min_cf_drop),
        )
        outputs["gt_fullgraph"] = _write_method_outputs(
            output_dir=gt_dir,
            method="gt_fullgraph",
            distance_type=distance_type,
            distance_name=distance_name,
            thresholds=thresholds,
            details=gt_details,
            summaries=gt_summaries,
            total_parents=len(parents),
            total_candidates=len(gt_candidates),
        )
        all_summaries.extend(gt_summaries)

    combined_dir = ensure_directory(output / "combined")
    _write_csv(combined_dir / "combined_threshold_summary.csv", all_summaries, DISTANCE_SUMMARY_FIELDS)
    _write_json(combined_dir / "combined_threshold_summary.json", {"threshold_summary": all_summaries})
    _save_figures(all_summaries, combined_dir / "figures", f"CCRCov {distance_name}")
    elapsed = time.time() - started
    (combined_dir / "combined_report.md").write_text(
        "# CCRCov Distance Evaluation\n\n"
        f"- distance_name: {distance_name}\n"
        f"- distance_type: {distance_type}\n"
        f"- elapsed_sec: {elapsed:.2f}\n"
        "- fullgraph baseline distance source: learned/embedding provider, not NetworkX GED by default.\n\n"
        "See `combined_threshold_summary.csv` and per-method reports for metrics.\n",
        encoding="utf-8",
    )
    print(f"[TASK_DONE] output_root={output} elapsed_sec={elapsed:.2f}", flush=True)
    return {"output_root": str(output), "outputs": outputs, "combined_summary": str(combined_dir / "combined_threshold_summary.csv")}
