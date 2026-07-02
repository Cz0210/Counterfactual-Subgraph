#!/usr/bin/env python3
"""Evaluate exported GlobalGCE outputs under the project CCRCov protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_adapter import (  # noqa: E402
    compute_globalgce_coverage_redundancy,
    compute_globalgce_structural_redundancy,
    globalgce_rule_to_action,
    graph_record_to_networkx,
    label_alignment_audit,
    load_globalgce_cfs,
    load_globalgce_processed_graphs,
    load_globalgce_rules,
)
from src.eval.close_counterfactual_coverage import (  # noqa: E402
    normalized_networkx_ged_distance,
    predict_with_teacher,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional.
    nx = None


DEFAULT_THRESHOLDS = "0.05,0.10,0.20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GlobalGCE exported outputs with unified CCRCov metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-mode", choices=["native-cf", "rule-action"], default="native-cf")
    parser.add_argument("--run-root", default="outputs/hpc/globalgce/aids_official_top30")
    parser.add_argument("--export-dir", default="outputs/hpc/globalgce/aids_official_top30_exported")
    parser.add_argument("--dataset", default="AIDS")
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--teacher-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--config", default=None, help="Ignored compatibility hook for HPC wrappers.")
    parser.add_argument("--set", action="append", default=[], help="Ignored compatibility hook for HPC wrappers.")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def safe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def load_manifest_commit(export_dir: Path, run_root: Path) -> str | None:
    commit = read_text_if_exists(run_root / "globalgce_source_commit.txt")
    if commit:
        return commit
    manifest_path = export_dir / "globalgce_files_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return manifest.get("globalgce_commit")


def filter_parent_records(
    records: list[dict[str, Any]],
    *,
    label: int,
    max_graphs: int | None,
) -> list[dict[str, Any]]:
    filtered = [record for record in records if record.get("label") == int(label)]
    if max_graphs is not None:
        return filtered[: int(max_graphs)]
    return filtered


def evaluate_native_cf(args: argparse.Namespace, thresholds: list[float], out_dir: Path) -> dict[str, Any]:
    if not args.teacher_path:
        raise ValueError("--teacher-path is required for native-cf evaluation")

    run_root = Path(args.run_root).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    dataset_dir = run_root / "GlobalGCE_src" / "datasets" / args.dataset
    parents_all = load_globalgce_processed_graphs(dataset_dir)
    parents = filter_parent_records(parents_all, label=args.label, max_graphs=args.max_graphs)
    parent_by_idx = {int(record["graph_idx"]): record for record in parents}
    cfs = load_globalgce_cfs(export_dir / "globalgce_cfs_graphs.jsonl")

    teacher = TeacherSemanticScorer(args.teacher_path)
    before_cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    valid_parent_ids: set[int] = set()
    flipped_parent_ids: set[int] = set()

    for cf_index, cf in enumerate(cfs):
        graph_idx = cf.get("graph_idx")
        try:
            graph_idx_int = int(graph_idx)
        except Exception:
            graph_idx_int = -1
        parent = parent_by_idx.get(graph_idx_int)
        row: dict[str, Any] = {
            "method": "GlobalGCE",
            "eval_mode": "native-cf",
            "graph_idx": graph_idx_int,
            "cf_index": cf.get("cf_index", cf_index),
            "parent_internal_label": parent.get("label") if parent else None,
            "parent_smiles": parent.get("smiles") if parent else None,
            "cf_smiles": cf.get("smiles"),
            "valid": False,
            "pred_before": None,
            "pred_after": None,
            "p_before": None,
            "p_after": None,
            "cf_drop": None,
            "cf_flip": False,
            "cost": None,
            "error": None,
        }
        if parent is None:
            row["error"] = "parent_not_in_label_subset_or_missing"
            rows.append(row)
            continue
        if not parent.get("smiles"):
            row["error"] = f"parent_smiles_invalid:{parent.get('smiles_error')}"
            rows.append(row)
            continue
        if not cf.get("smiles"):
            row["error"] = f"cf_smiles_invalid:{cf.get('smiles_error')}"
            rows.append(row)
            continue

        if graph_idx_int not in before_cache:
            before_cache[graph_idx_int] = predict_with_teacher(teacher, parent["smiles"], int(args.label))
        before = before_cache[graph_idx_int]
        after = predict_with_teacher(teacher, cf["smiles"], int(args.label))
        if not before.get("ok"):
            row["error"] = f"before_teacher_failed:{before.get('error')}"
            rows.append(row)
            continue
        if not after.get("ok"):
            row["error"] = f"after_teacher_failed:{after.get('error')}"
            rows.append(row)
            continue

        cost = normalized_networkx_ged_distance(parent["smiles"], cf["smiles"], timeout=2.0)
        row.update(
            {
                "valid": cost is not None,
                "pred_before": before.get("pred_label"),
                "pred_after": after.get("pred_label"),
                "p_before": before.get("p_label"),
                "p_after": after.get("p_label"),
                "cf_drop": (
                    float(before["p_label"]) - float(after["p_label"])
                    if before.get("p_label") is not None and after.get("p_label") is not None
                    else None
                ),
                "cf_flip": bool(after.get("pred_label") != int(args.label)),
                "cost": cost,
                "error": None if cost is not None else "ged_failed_or_unavailable",
            }
        )
        if row["valid"]:
            valid_parent_ids.add(graph_idx_int)
        if row["valid"] and row["cf_flip"]:
            flipped_parent_ids.add(graph_idx_int)
        rows.append(row)

    num_graphs = len(parents)
    ccrc = {}
    best_costs_by_threshold: dict[float, list[float]] = {}
    covered_drop_values: list[float] = []
    for threshold in thresholds:
        covered: set[int] = set()
        costs: list[float] = []
        for graph_idx in parent_by_idx:
            candidates = [
                row
                for row in rows
                if row["graph_idx"] == graph_idx
                and row.get("valid")
                and row.get("cost") is not None
                and float(row["cost"]) <= threshold
                and row.get("cf_flip")
            ]
            if candidates:
                best = sorted(candidates, key=lambda item: (float(item["cost"]), -float(item.get("cf_drop") or 0.0)))[0]
                covered.add(graph_idx)
                costs.append(float(best["cost"]))
                if best.get("cf_drop") is not None and threshold == max(thresholds):
                    covered_drop_values.append(float(best["cf_drop"]))
        ccrc[f"CCRCov@{threshold:.2f}"] = safe_rate(len(covered), num_graphs)
        best_costs_by_threshold[threshold] = costs

    max_threshold = max(thresholds) if thresholds else 0.20
    summary = {
        "method": "GlobalGCE",
        "dataset": args.dataset,
        "label": int(args.label),
        "eval_mode": "native-cf",
        "k": int(args.k),
        "num_graphs": num_graphs,
        "num_rules": len(read_jsonl(export_dir / "globalgce_rules.jsonl")),
        "num_cfs": len(cfs),
        "SuppCov@K": None,
        "CFDrop": safe_mean(covered_drop_values),
        "FlipRate": safe_rate(len(flipped_parent_ids), len(valid_parent_ids)),
        "CostMean": safe_mean(best_costs_by_threshold.get(max_threshold, [])),
        "CostMedian": safe_median(best_costs_by_threshold.get(max_threshold, [])),
        "StructRed": None,
        "CovRed": None,
        "valid_rate": safe_rate(sum(1 for row in rows if row.get("valid")), len(cfs)),
        "label_alignment_warning": label_alignment_audit()["label_alignment_warning"],
        "label_alignment_audit": label_alignment_audit(),
        "source_run_root": str(run_root),
        "source_export_dir": str(export_dir),
        "globalgce_commit": load_manifest_commit(export_dir, run_root),
        "project_git_commit": git_commit(),
    }
    summary.update(ccrc)

    write_csv(
        out_dir / "per_graph.csv",
        rows,
        [
            "method",
            "eval_mode",
            "graph_idx",
            "cf_index",
            "parent_internal_label",
            "parent_smiles",
            "cf_smiles",
            "valid",
            "pred_before",
            "pred_after",
            "p_before",
            "p_after",
            "cf_drop",
            "cf_flip",
            "cost",
            "error",
        ],
    )
    write_csv(out_dir / "per_rule.csv", [], ["rule_id", "match_count", "covered_graphs", "note"])
    return summary


def node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("label") == right.get("label")


def edge_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("label") == right.get("label")


def evaluate_rule_action(args: argparse.Namespace, thresholds: list[float], out_dir: Path) -> dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    dataset_dir = run_root / "GlobalGCE_src" / "datasets" / args.dataset
    parents_all = load_globalgce_processed_graphs(dataset_dir)
    parents = filter_parent_records(parents_all, label=args.label, max_graphs=args.max_graphs)
    rules = load_globalgce_rules(export_dir / "globalgce_rules.jsonl")
    if args.k > 0:
        rules = rules[: args.k]

    parent_graphs = {int(record["graph_idx"]): graph_record_to_networkx(record) for record in parents}
    rule_rows: list[dict[str, Any]] = []
    per_graph_rows: list[dict[str, Any]] = []
    rule_cover_sets: dict[int, set[int]] = {}

    for rule in rules:
        rule_id = int(rule.get("rule_id", len(rule_rows)))
        lhs_graph = graph_record_to_networkx(rule.get("lhs") or {})
        matched: set[int] = set()
        reason = ""
        if nx is None:
            reason = "networkx_unavailable"
        elif lhs_graph is None or lhs_graph.number_of_nodes() == 0:
            reason = "lhs_graph_unavailable_or_empty"
        else:
            for graph_idx, parent_graph in parent_graphs.items():
                if parent_graph is None:
                    continue
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    parent_graph,
                    lhs_graph,
                    node_match=node_match,
                    edge_match=edge_match,
                )
                if matcher.subgraph_is_isomorphic():
                    matched.add(graph_idx)
        rule_cover_sets[rule_id] = matched
        action = globalgce_rule_to_action(rule)
        rule_rows.append(
            {
                "rule_id": rule_id,
                "match_count": len(matched),
                "covered_graphs": json.dumps(sorted(matched)),
                "rule_action_supported": action["rule_action_supported"],
                "unsupported_reason": action["unsupported_reason"],
                "note": reason,
            }
        )

    covered_by_any: set[int] = set()
    for covered in rule_cover_sets.values():
        covered_by_any.update(covered)
    for parent in parents:
        graph_idx = int(parent["graph_idx"])
        matched_rules = [rule_id for rule_id, covered in rule_cover_sets.items() if graph_idx in covered]
        per_graph_rows.append(
            {
                "method": "GlobalGCE",
                "eval_mode": "rule-action",
                "graph_idx": graph_idx,
                "parent_internal_label": parent.get("label"),
                "parent_smiles": parent.get("smiles"),
                "matched_rule_ids": json.dumps(matched_rules),
                "suppcov_match": bool(matched_rules),
                "error": "",
            }
        )

    summary = {
        "method": "GlobalGCE",
        "dataset": args.dataset,
        "label": int(args.label),
        "eval_mode": "rule-action",
        "k": int(args.k),
        "num_graphs": len(parents),
        "num_rules": len(rules),
        "num_cfs": len(read_jsonl(export_dir / "globalgce_cfs_graphs.jsonl")),
        "SuppCov@K": safe_rate(len(covered_by_any), len(parents)),
        "CFDrop": None,
        "FlipRate": None,
        "CostMean": None,
        "CostMedian": None,
        "StructRed": compute_globalgce_structural_redundancy(rules),
        "CovRed": compute_globalgce_coverage_redundancy(rule_cover_sets),
        "valid_rate": None,
        "rule_action_supported": False,
        "rule_action_unsupported_reason": "Safe RHS replacement is not implemented.",
        "label_alignment_warning": label_alignment_audit()["label_alignment_warning"],
        "label_alignment_audit": label_alignment_audit(),
        "source_run_root": str(run_root),
        "source_export_dir": str(export_dir),
        "globalgce_commit": load_manifest_commit(export_dir, run_root),
        "project_git_commit": git_commit(),
    }
    for threshold in thresholds:
        summary[f"CCRCov@{threshold:.2f}"] = None

    write_csv(
        out_dir / "per_graph.csv",
        per_graph_rows,
        [
            "method",
            "eval_mode",
            "graph_idx",
            "parent_internal_label",
            "parent_smiles",
            "matched_rule_ids",
            "suppcov_match",
            "error",
        ],
    )
    write_csv(
        out_dir / "per_rule.csv",
        rule_rows,
        ["rule_id", "match_count", "covered_graphs", "rule_action_supported", "unsupported_reason", "note"],
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "GlobalGCE Unified Evaluation Report",
        "",
        f"method: {summary.get('method')}",
        f"dataset: {summary.get('dataset')}",
        f"label: {summary.get('label')}",
        f"eval_mode: {summary.get('eval_mode')}",
        f"k: {summary.get('k')}",
        f"num_graphs: {summary.get('num_graphs')}",
        f"num_rules: {summary.get('num_rules')}",
        f"num_cfs: {summary.get('num_cfs')}",
        f"SuppCov@K: {summary.get('SuppCov@K')}",
        f"CCRCov@0.05: {summary.get('CCRCov@0.05')}",
        f"CCRCov@0.10: {summary.get('CCRCov@0.10')}",
        f"CCRCov@0.20: {summary.get('CCRCov@0.20')}",
        f"CFDrop: {summary.get('CFDrop')}",
        f"FlipRate: {summary.get('FlipRate')}",
        f"CostMean: {summary.get('CostMean')}",
        f"CostMedian: {summary.get('CostMedian')}",
        f"StructRed: {summary.get('StructRed')}",
        f"CovRed: {summary.get('CovRed')}",
        f"valid_rate: {summary.get('valid_rate')}",
        "",
        "Label alignment warning:",
        str(summary.get("label_alignment_warning")),
        "",
        f"source_run_root: {summary.get('source_run_root')}",
        f"source_export_dir: {summary.get('source_export_dir')}",
        f"globalgce_commit: {summary.get('globalgce_commit')}",
        f"project_git_commit: {summary.get('project_git_commit')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = PROJECT_ROOT / "outputs" / "hpc" / "eval" / "globalgce" / "aids_official_top30" / f"label{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_mode == "native-cf":
        summary = evaluate_native_cf(args, thresholds, out_dir)
    else:
        summary = evaluate_rule_action(args, thresholds, out_dir)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_report(out_dir / "report.txt", summary)
    print("[GLOBALGCE_EVAL]")
    print(f"output_dir={out_dir}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
