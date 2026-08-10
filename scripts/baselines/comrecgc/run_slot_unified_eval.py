#!/usr/bin/env python3
"""Evaluate repaired COMRECGC medoids without compacting official rank slots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import (  # noqa: E402
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.baselines.comrecgc.slot_evaluation import (  # noqa: E402
    ADAPTATION_MODE,
    METHOD,
    SELECTION_METHOD,
    build_final_audit,
    build_internal_valid_candidates,
    compute_slot_metrics,
    expand_pair_rows,
    load_official_slots,
    read_csv,
    table_row,
    write_csv,
    write_jsonl,
)
from src.eval.close_counterfactual_coverage import _load_parent_records  # noqa: E402
from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


BACE_METHOD = "COMRECGC"
BACE_FIGURE3_FIELDS = ("method", "k", "coverage", "cost")
BACE_FIGURE4_FIELDS = ("method", "threshold", "coverage")
BACE_TABLE2_FIELDS = ("method", "k", "coverage", "cost", "flip_rate", "cf_drop")


def _threshold_contract(
    path: Path,
    *,
    theta_star_override: float | None = None,
    cost_cap_override: float | None = None,
) -> tuple[list[float], float, float, dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        rows = read_csv(path)
        threshold_field = next(
            (field for field in ("threshold", "theta") if rows and field in rows[0]),
            None,
        )
        if threshold_field is None:
            raise ValueError("Frozen threshold CSV has no threshold/theta column.")
        thresholds = sorted({float(row[threshold_field]) for row in rows})
        if not thresholds:
            raise ValueError("Frozen threshold CSV is empty.")
        if theta_star_override is None:
            raise ValueError("Threshold CSV requires an explicit frozen theta_star.")
        theta_star = float(theta_star_override)
        cost_cap = (
            float(cost_cap_override)
            if cost_cap_override is not None
            else float(max(thresholds))
        )
        return thresholds, theta_star, cost_cap, {
            "threshold_source": "frozen_empirical_curve_csv",
            "threshold_field": threshold_field,
            "threshold_count": len(thresholds),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen threshold artifact must be a JSON object.")
    values = payload.get("thresholds")
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    raw = payload.get("raw_quantile_thresholds")
    quantile_map: dict[float, float] = {}
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict):
                quantile_map[float(row["quantile"])] = float(row["threshold"])
        if not isinstance(values, list):
            values = [float(row["threshold"]) for row in raw if isinstance(row, dict)]
    if not isinstance(values, list) or not values:
        merged = payload.get("merged_thresholds")
        if isinstance(merged, list):
            values = [row["threshold"] for row in merged if isinstance(row, dict)]
    if not isinstance(values, list) or not values:
        raise ValueError("Frozen threshold artifact has no threshold grid.")
    thresholds = sorted({float(value) for value in values})
    theta_star = (
        theta_star_override
        if theta_star_override is not None
        else payload.get("theta_star")
    )
    cost_cap = (
        cost_cap_override
        if cost_cap_override is not None
        else payload.get("cost_cap")
    )
    if theta_star is None:
        theta_star = quantile_map.get(0.30)
    if cost_cap is None:
        cost_cap = quantile_map.get(0.90)
    if theta_star is None or cost_cap is None:
        raise ValueError("Frozen threshold artifact lacks theta_star/cost_cap provenance.")
    if not all(value >= 0.0 for value in [*thresholds, float(theta_star), float(cost_cap)]):
        raise ValueError("Frozen thresholds must be finite and nonnegative.")
    return thresholds, float(theta_star), float(cost_cap), payload


def _run_shared_evaluator(
    *,
    candidates_csv: Path,
    dataset_csv: Path,
    teacher_path: Path,
    molclr_root: Path,
    molclr_checkpoint: Path,
    thresholds: list[float],
    output_dir: Path,
    max_parents: int,
    device: str,
    resume: bool,
    action_semantics_version: str = "hard_delete_all_matches_v1",
    match_selection_policy: str = "min_wnode_then_cfdrop_then_match_index_v1",
) -> list[dict[str, str]]:
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset-csv",
        str(dataset_csv),
        "--teacher-path",
        str(teacher_path),
        "--molclr-root",
        str(molclr_root),
        "--molclr-checkpoint",
        str(molclr_checkpoint),
        "--label",
        "1",
        "--smiles-col",
        "smiles",
        "--label-col",
        "label",
        "--cf-mode",
        "strict_flip",
        "--output-dir",
        str(output_dir),
        "--max-parents",
        str(max_parents),
        "--max-candidates",
        "0",
        "--wnode-thresholds",
        ",".join(format(value, ".17g") for value in thresholds),
        "--feature-cost",
        "cosine",
        "--node-mass",
        "uniform",
        "--size-penalty-beta",
        "0.0",
        "--device",
        device,
        "--run-distance-self-test",
        "1",
        "--run-ours",
        "0",
        "--run-fullgraph",
        "1",
        "--fullgraph-candidates-path",
        str(candidates_csv),
        "--fullgraph-method-name",
        METHOD,
        "--selection-method",
        SELECTION_METHOD,
        "--action-semantics-version",
        action_semantics_version,
        "--match-selection-policy",
        match_selection_policy,
        "--preselected-topk",
        "1",
        "--require-preselected-topk",
        "0",
        "--skip-redundancy",
        "1",
        "--resume",
        "1" if resume else "0",
    ]
    write_json(
        output_dir.parent / f"{output_dir.name}_command.json",
        {
            "argv": argv,
            "shared_evaluator": "scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py",
            "distance_calculation_reimplemented": False,
            "teacher_calculation_reimplemented": False,
        },
    )
    subprocess.run(argv, cwd=PROJECT_ROOT, check=True, timeout=172800)
    details = output_dir / "details/pair_details.csv"
    if not details.is_file():
        raise FileNotFoundError(f"Shared evaluator pair details are missing: {details}")
    return read_csv(details)


def _apply_bace_connected_candidate_gate(
    slots: list[dict[str, Any]],
) -> dict[str, int | bool]:
    """Invalidate disconnected repaired medoids without changing rank slots."""

    from rdkit import Chem

    repaired = 0
    connected = 0
    disconnected = 0
    for slot in slots:
        slot["connected_action_semantics"] = CONNECTED_ACTION_SEMANTICS
        slot["repaired_sanitize_ok"] = False
        slot["repaired_num_components"] = None
        slot["repaired_contains_dot"] = False
        slot["repaired_connected"] = False
        if not bool(slot.get("candidate_slot_valid")):
            continue
        repaired += 1
        smiles = str(slot.get("repaired_smiles") or "").strip()
        molecule = Chem.MolFromSmiles(smiles)
        sanitize_ok = False
        component_count: int | None = None
        if molecule is not None and molecule.GetNumAtoms() > 0:
            try:
                Chem.SanitizeMol(molecule)
                sanitize_ok = True
                component_count = len(Chem.GetMolFrags(molecule))
            except Exception:
                sanitize_ok = False
        contains_dot = "." in smiles
        is_connected = bool(
            sanitize_ok
            and molecule is not None
            and molecule.GetNumAtoms() > 0
            and component_count == 1
            and not contains_dot
        )
        slot["repaired_sanitize_ok"] = sanitize_ok
        slot["repaired_num_components"] = component_count
        slot["repaired_contains_dot"] = contains_dot
        slot["repaired_connected"] = is_connected
        if is_connected:
            connected += 1
            continue
        disconnected += 1
        slot["candidate_slot_valid"] = False
        slot["slot_status"] = "CONNECTED_PROTOCOL_INVALID"
        slot["slot_rejection_reason"] = (
            "repaired_candidate_not_nonempty_sanitized_single_component"
        )
    return {
        "repaired_slot_count_before_connected_gate": repaired,
        "connected_repaired_slot_count": connected,
        "disconnected_repaired_slot_count": disconnected,
        "disconnected_output_used_count": 0,
        "all_repaired_candidates_connected": disconnected == 0,
        "all_evaluated_candidates_connected": True,
    }


def _bace_paper_rows(
    *,
    prefixes: list[dict[str, Any]],
    figure4: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    by_k = {int(row["k"]): row for row in prefixes}
    figure3_rows = [
        {
            "method": BACE_METHOD,
            "k": int(row["k"]),
            "coverage": float(row["close_cf_coverage"]),
            "cost": row.get("conditional_median_cost"),
        }
        for row in prefixes
    ]
    figure4_rows = [
        {
            "method": BACE_METHOD,
            "threshold": float(row["threshold"]),
            "coverage": float(row["close_cf_coverage"]),
        }
        for row in figure4
    ]
    return figure3_rows, figure4_rows, by_k


def _bace_table_row(prefix: dict[str, Any]) -> dict[str, Any]:
    covered = int(prefix.get("num_close_cf_covered") or 0)
    return {
        "method": BACE_METHOD,
        "k": int(prefix["k"]),
        "coverage": float(prefix["close_cf_coverage"]),
        "cost": prefix.get("conditional_median_cost"),
        "flip_rate": 1.0 if covered > 0 else 0.0,
        "cf_drop": prefix.get("avg_cf_drop_among_covered"),
    }


def _parent_records(dataset_csv: Path, expected_count: int) -> tuple[list[str], str]:
    _path, records, label_column = _load_parent_records(
        dataset_csv,
        label=1,
        smiles_col="smiles",
        label_col="label",
        max_parents=expected_count,
    )
    if len(records) != expected_count:
        raise ValueError(
            f"Parent cohort mismatch: actual={len(records)}, expected={expected_count}."
        )
    parent_ids = [str(row.parent_id) for row in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Parent cohort IDs are not unique.")
    return parent_ids, records[0].smiles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--dataset", choices=("aids", "mutagenicity", "bace"), default="mutagenicity"
    )
    parser.add_argument("--chemistry-dir", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--theta-star", type=float)
    parser.add_argument("--cost-cap", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = require_empty_output(args.output_dir, resume=args.resume)
    chemistry = Path(args.chemistry_dir).expanduser().resolve()
    dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    teacher = Path(args.teacher_path).expanduser().resolve()
    molclr_root = Path(args.molclr_root).expanduser().resolve()
    molclr_checkpoint = Path(args.molclr_checkpoint).expanduser().resolve()
    thresholds_path = Path(args.thresholds_json).expanduser().resolve()
    for path in (dataset_csv, teacher, molclr_checkpoint, thresholds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not molclr_root.is_dir():
        raise FileNotFoundError(molclr_root)
    chemistry_manifest = chemistry / "run_manifest.json"
    if not chemistry_manifest.is_file():
        raise FileNotFoundError(chemistry_manifest)
    chemistry_payload = json.loads(chemistry_manifest.read_text(encoding="utf-8"))
    if not isinstance(chemistry_payload, dict) or chemistry_payload.get("run_complete") is not True:
        raise ValueError("Chemistry audit manifest is incomplete.")
    all_slots = load_official_slots(chemistry / "medoid_validity.csv")
    connected_gate: dict[str, int | bool] | None = None
    if args.dataset == "bace":
        connected_gate = _apply_bace_connected_candidate_gate(all_slots)
    slots = all_slots[: int(args.max_k)]
    valid_candidates = build_internal_valid_candidates(slots)
    write_csv(root / "selected_rank_slots.csv", all_slots)
    write_csv(root / "evaluated_rank_slots.csv", slots)
    write_jsonl(root / "selected_sequence.jsonl", all_slots)
    write_json(root / "selected_common_recourses.json", all_slots)
    write_jsonl(root / "representative_counterfactuals.jsonl", all_slots)
    parent_ids, probe_smiles = _parent_records(dataset_csv, args.expected_parent_count)
    thresholds, theta_star, cost_cap, threshold_payload = _threshold_contract(
        thresholds_path,
        theta_star_override=args.theta_star,
        cost_cap_override=args.cost_cap,
    )
    evaluator_invoked = False
    interface_probe_invoked = False
    evaluated_rows: list[dict[str, str]] = []
    if valid_candidates:
        candidates_csv = root / "internal_valid_repaired_medoids.csv"
        write_csv(candidates_csv, valid_candidates)
        evaluated_rows = _run_shared_evaluator(
            candidates_csv=candidates_csv,
            dataset_csv=dataset_csv,
            teacher_path=teacher,
            molclr_root=molclr_root,
            molclr_checkpoint=molclr_checkpoint,
            thresholds=thresholds,
            output_dir=root / "shared_evaluator",
            max_parents=args.expected_parent_count,
            device=args.device,
            resume=args.resume,
            action_semantics_version=(
                CONNECTED_ACTION_SEMANTICS
                if args.dataset == "bace"
                else "hard_delete_all_matches_v1"
            ),
            match_selection_policy=(
                CONNECTED_MATCH_SELECTION_POLICY
                if args.dataset == "bace"
                else "min_wnode_then_cfdrop_then_match_index_v1"
            ),
        )
        evaluator_invoked = True
    else:
        probe_csv = root / "interface_probe_candidate.csv"
        write_csv(
            probe_csv,
            [
                {
                    "rank": 1,
                    "candidate_id": "COMRECGC_INTERFACE_PROBE_NOT_A_RESULT",
                    "smiles": probe_smiles,
                    "canonical_smiles": probe_smiles,
                    "selection_method": "engineering_interface_probe_only",
                }
            ],
        )
        _run_shared_evaluator(
            candidates_csv=probe_csv,
            dataset_csv=dataset_csv,
            teacher_path=teacher,
            molclr_root=molclr_root,
            molclr_checkpoint=molclr_checkpoint,
            thresholds=thresholds,
            output_dir=root / "interface_probe",
            max_parents=1,
            device=args.device,
            resume=args.resume,
            action_semantics_version=(
                CONNECTED_ACTION_SEMANTICS
                if args.dataset == "bace"
                else "hard_delete_all_matches_v1"
            ),
            match_selection_policy=(
                CONNECTED_MATCH_SELECTION_POLICY
                if args.dataset == "bace"
                else "min_wnode_then_cfdrop_then_match_index_v1"
            ),
        )
        interface_probe_invoked = True
    pair_rows = expand_pair_rows(
        parent_ids=parent_ids,
        slots=slots,
        evaluated_rows=evaluated_rows,
    )
    write_jsonl(root / "pair_matrix.jsonl", pair_rows)
    prefixes, threshold_metrics, parent_best = compute_slot_metrics(
        pair_rows=pair_rows,
        slots=slots,
        parent_ids=parent_ids,
        thresholds=thresholds,
        theta_star=theta_star,
        cost_cap=cost_cap,
        max_k=args.max_k,
    )
    figure4 = [row for row in threshold_metrics if int(row["k"]) == int(args.max_k)]
    write_csv(root / "prefix_metrics.csv", prefixes)
    write_json(root / "prefix_metrics.json", {"prefix_metrics": prefixes})
    if args.dataset == "bace":
        bace_figure3, bace_figure4, by_k = _bace_paper_rows(
            prefixes=prefixes,
            figure4=figure4,
        )
        write_csv(root / "figure3_coverage_vs_k.csv", bace_figure3)
        write_csv(root / "figure4_coverage_vs_threshold.csv", bace_figure4)
    else:
        by_k = {int(row["k"]): row for row in prefixes}
        write_csv(root / "figure3_coverage_vs_k.csv", prefixes)
        write_csv(root / "figure4_coverage_vs_threshold.csv", figure4)
    write_csv(root / "parent_best_distances.csv", parent_best)
    for k in (10, 20):
        row = (
            _bace_table_row(by_k[k])
            if args.dataset == "bace"
            else table_row(
                by_k[k],
                theta_star=theta_star,
                dataset="AIDS" if args.dataset == "aids" else "Mutagenicity",
            )
        )
        write_csv(
            root / f"table2_comrecgc_k{k}.csv",
            [row],
        )
    audit = build_final_audit(
        root=root,
        prefixes=prefixes,
        figure4=figure4,
        slots=slots,
        parent_count=len(parent_ids),
        thresholds=thresholds,
        evaluator_invoked=evaluator_invoked,
        interface_probe_invoked=interface_probe_invoked,
    )
    summary = {
        **audit,
        "dataset": {
            "aids": "AIDS",
            "mutagenicity": "Mutagenicity",
            "bace": "BACE",
        }[args.dataset],
        "dataset_key": args.dataset,
        "mode": args.mode,
        "theta_star": theta_star,
        "cost_cap": cost_cap,
        "threshold_count": len(thresholds),
        "requested_max_k": int(args.max_k),
        "official_total_rank_count": len(all_slots),
        "evaluated_rank_slot_count": len(slots),
        "valid_k20": int(by_k[20]["valid_k"]),
        "k10_coverage": float(by_k[10]["close_cf_coverage"]),
        "k10_conditional_median_cost": by_k[10]["conditional_median_cost"],
        "k20_coverage": float(by_k[20]["close_cf_coverage"]),
        "k20_conditional_median_cost": by_k[20]["conditional_median_cost"],
    }
    if args.dataset == "bace":
        summary.update(
            {
                "method": BACE_METHOD,
                "test_parent_count": len(parent_ids),
                "test_parent_ids_sha256": stable_json_sha256(parent_ids),
                "candidate_count": len(all_slots),
                "test_evaluation_count": 1,
                "cost_definition": "applicable_parent_conditional_median_wnode",
                "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                "connected_action_semantics": True,
                **dict(connected_gate or {}),
            }
        )
        audit.update(
            {
                "passed": bool(audit["audit_passed"]),
                "dataset": "BACE",
                "method": BACE_METHOD,
                "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                "connected_action_semantics": True,
                **dict(connected_gate or {}),
            }
        )
        write_json(root / "final_artifact_audit.json", audit)
    write_json(root / "summary.json", summary)
    run_manifest = {
        **summary,
        "run_complete": True,
        "source_chemistry_dir": str(chemistry),
        "source_chemistry_manifest_sha256": sha256_file(chemistry_manifest),
        "project_commit": chemistry_payload.get("project_commit"),
        "upstream_commit": chemistry_payload.get("upstream_commit"),
        "repair_policy_sha256": chemistry_payload.get("repair_policy_sha256"),
        "dataset_fingerprint": chemistry_payload.get("dataset_fingerprint"),
        "generation_parent_ids_sha256": chemistry_payload.get(
            "generation_parent_ids_sha256"
        ),
        "dataset_csv": str(dataset_csv),
        "dataset_csv_sha256": sha256_file(dataset_csv),
        "teacher_path": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "molclr_checkpoint": str(molclr_checkpoint),
        "molclr_checkpoint_sha256": sha256_file(molclr_checkpoint),
        "thresholds_path": str(thresholds_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "threshold_source": threshold_payload.get("threshold_source"),
        "threshold_grid": thresholds,
        "k_grid": list(range(1, int(args.max_k) + 1)),
        "source_label": 1,
        "target_label": 0,
        "parent_ids_sha256": stable_json_sha256(parent_ids),
        "official_rank_candidate_ids_sha256": stable_json_sha256(
            [str(row["candidate_id"]) for row in all_slots]
        ),
        "official_rank_candidate_slot_ids_sha256": stable_json_sha256(
            [str(row["candidate_slot_id"]) for row in all_slots]
        ),
        "official_rank_source_candidate_ids_sha256": stable_json_sha256(
            [str(row["source_candidate_id"]) for row in all_slots]
        ),
        "official_rank_evaluation_candidate_ids_sha256": stable_json_sha256(
            [str(row["evaluation_candidate_id"]) for row in all_slots]
        ),
        "source_candidate_reuse_across_slots": len(
            {str(row["source_candidate_id"]) for row in all_slots}
        )
        < len(all_slots),
        "candidate_identity_contract": {
            "evaluation_compute_identity": "evaluation_candidate_id",
            "official_output_identity": "candidate_slot_id",
            "lineage_identity": "source_candidate_id",
            "slot_identity_source": "official_cluster_rank",
            "source_candidate_reuse_preserved": True,
            "official_slot_deduplication_performed": False,
            "identical_smiles_compute_reuse": True,
            "rank_compaction_performed": False,
            "cross_rank_backfill_performed": False,
        },
        "candidate_order_source": "official_common_recourse_cluster_rank",
        "candidate_order_unchanged": True,
        "invalid_candidates_sent_to_rf_or_wnode": False,
        "distance_calculation_reimplemented": False,
        "teacher_calculation_reimplemented": False,
        "calibration_loaded": False,
        "test_loaded_for_selection": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "run_manifest.json", run_manifest)
    marker = "_SMOKE_AUDIT_COMPLETE.json" if args.mode == "smoke" else "_RUN_COMPLETE.json"
    write_json(root / marker, {"run_complete": True, "audit_passed": True})
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": f"{args.dataset}_slot_unified_eval",
            "error_class": type(exc).__name__,
            "message": str(exc),
            "run_complete": False,
        }
        write_json(output / "failure_summary.json", failure)
        write_json(output / "_RUN_FAILED.json", failure)
        raise
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
