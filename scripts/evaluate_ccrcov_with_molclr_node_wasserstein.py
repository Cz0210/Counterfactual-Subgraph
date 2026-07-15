#!/usr/bin/env python3
"""Evaluate unified CCRCov with exact MolCLR node Wasserstein distance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_ccrcov_with_molclr_node_fgw import (  # noqa: E402
    _detail_fields,
    _env,
    _env_bool,
    _finite_distance,
    _fragment_identity,
    _mean,
    _median,
    _parse_float_list,
    _rate,
    _row_flip,
    _write_csv,
    _write_json,
    build_evaluation_row_audit,
    build_threshold_table,
    infer_fullgraph_method,
    validate_preselected_candidate_csv,
    validate_preselected_ours_directory,
)
from src.eval.ccrcov_distance_eval import (  # noqa: E402
    CF_MODES,
    GT_DIRECTORY_CANDIDATES,
    OURS_DIRECTORY_CANDIDATES,
    _evaluate_gt_fullgraph,
    _evaluate_ours,
    _parse_int_label,
)
from src.eval.close_counterfactual_coverage import (  # noqa: E402
    DETAIL_FIELDS,
    _load_candidate_records,
    _load_parent_records,
)
from src.eval.flip_semantics import (  # noqa: E402
    OLD_WEAK_FLIP_DEFINITION,
    TEACHER_STRICT_FLIP_DEFINITION,
    old_weak_flip,
)
from src.eval.greed_distance.pair_generation import GT_FULLGRAPH_FIELDS, OURS_FRAGMENT_FIELDS  # noqa: E402
from src.eval.molclr_node_embeddings import DEFAULT_NODE_EMB_CACHE_DIR  # noqa: E402
from src.eval.node_wasserstein_distance import (  # noqa: E402
    DEFAULT_WNODE_CACHE_DB,
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402
from src.utils.io import ensure_directory  # noqa: E402


DISTANCE_TYPE = "node_wasserstein"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_NAME = "molclr_node_wasserstein"
DEFAULT_QUANTILES = "0.05,0.10,0.20,0.30,0.50,0.70,0.90"
SUMMARY_FIELDS = [
    "method", "distance_type", "distance_line", "feature_cost", "node_mass",
    "size_penalty_beta", "solver", "threshold", "threshold_source", "quantile",
    "cf_mode", "main_ccrcov_uses", "teacher_strict_flip_definition",
    "old_weak_flip_definition", "old_weak_ccrcov_status", "num_parents",
    "num_teacher_target_parents", "num_candidates", "num_valid_pairs",
    "num_close_only_covered", "close_only_coverage", "num_close_cf_covered",
    "close_cf_coverage", "old_weak_num_close_cf_covered",
    "old_weak_close_cf_coverage", "avg_best_distance", "median_best_distance",
    "avg_cf_drop_among_covered", "flip_rate_among_covered", "total_pairs",
    "cache_hit_rate", "node_embedding_cache_hit_rate", "skip_redundancy",
    "candidate_set_preselected", "selection_performed_in_eval", "selection_method",
    "evaluation_row_unit", "num_unique_parent_candidate_pairs", "num_detail_rows",
    "num_valid_match_instances",
]


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = _detail_fields(list(rows))
    ensure_directory(path.parent)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else ("" if row.get(field) is None else row.get(field))
                    for field in fields
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _file_identity(value: str | Path | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    try:
        stat = path.stat()
        if path.is_dir():
            children = []
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if not child.is_file():
                    continue
                child_stat = child.stat()
                children.append({"name": child.name, "size": child_stat.st_size, "mtime_ns": child_stat.st_mtime_ns})
            return {"path": str(path), "kind": "directory", "files": children}
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}


def config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResumeCheckpoint:
    def __init__(self, output: Path, method_key: str, fingerprint: str, enabled: bool) -> None:
        self.partial_path = output / "details" / f"{method_key}.partial.csv"
        self.state_path = output / "details" / f"{method_key}.resume.json"
        self.fingerprint = fingerprint
        self.enabled = bool(enabled)

    def load(self) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
        if not self.enabled or (not self.partial_path.is_file() and not self.state_path.is_file()):
            return [], set()
        if not self.partial_path.is_file() or not self.state_path.is_file():
            raise ValueError(f"Incomplete resume checkpoint pair: {self.partial_path} / {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if str(state.get("config_fingerprint")) != self.fingerprint:
            raise ValueError(
                "Resume configuration does not match the saved partial checkpoint: "
                f"method={self.partial_path.stem}"
            )
        rows = _read_csv_rows(self.partial_path)
        completed = {
            (str(pair[0]), str(pair[1]))
            for pair in state.get("completed_pair_keys", [])
            if isinstance(pair, list) and len(pair) == 2
        }
        row_pairs = {(str(row.get("parent_id") or ""), str(row.get("candidate_id") or "")) for row in rows}
        if not completed.issubset(row_pairs):
            raise ValueError("Resume state lists completed pairs absent from the partial detail CSV.")
        # If a crash happened after replacing the CSV but before replacing the
        # state JSON, discard those uncommitted rows so recomputation cannot
        # create duplicate detail records.
        committed_rows = [
            row for row in rows
            if (str(row.get("parent_id") or ""), str(row.get("candidate_id") or "")) in completed
        ]
        return committed_rows, completed

    def save(self, rows: list[dict[str, Any]], completed: set[tuple[str, str]]) -> None:
        _atomic_write_csv(self.partial_path, rows)
        _atomic_write_json(
            self.state_path,
            {
                "config_fingerprint": self.fingerprint,
                "completed_pair_keys": [list(pair) for pair in sorted(completed)],
                "num_completed_pairs": len(completed),
                "num_detail_rows": len(rows),
                "distance_recomputed_for_completed_pairs": False,
                "updated_at_epoch_seconds": time.time(),
            },
        )


def _best_row_for_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for row in rows:
        distance = _finite_distance(row)
        if distance is None or distance > threshold:
            continue
        label = _parse_int_label(row.get("label"))
        pred_before = _parse_int_label(row.get("pred_before"))
        pred_after = _parse_int_label(row.get("pred_after"))
        if label is None or pred_before != label or pred_after == label:
            continue
        if best is None or distance < best[0]:
            best = (distance, row)
    return best[1] if best else None


def summarize_method(
    *,
    method: str,
    details: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    config: MolCLRNodeWassersteinConfig,
    cf_mode: str,
    cache_hit_rate: float,
    node_embedding_cache_hit_rate: float,
    skip_redundancy: bool,
    group_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        by_parent.setdefault(str(row.get("parent_id") or ""), []).append(row)
    teacher_target = {
        parent_id
        for parent_id, rows in by_parent.items()
        if any(_parse_int_label(row.get("pred_before")) == _parse_int_label(row.get("label")) for row in rows)
    }
    valid_pairs = sum(_finite_distance(row) is not None for row in details)
    output: list[dict[str, Any]] = []
    for threshold_row in threshold_rows:
        threshold = float(threshold_row["threshold"])
        if not math.isfinite(threshold):
            continue
        close_only = {
            parent_id for parent_id, rows in by_parent.items()
            if any((distance := _finite_distance(row)) is not None and distance <= threshold for row in rows)
        }
        best_rows = [row for rows in by_parent.values() if (row := _best_row_for_threshold(rows, threshold)) is not None]
        close_cf = {str(row.get("parent_id") or "") for row in best_rows}
        weak = {
            parent_id for parent_id, rows in by_parent.items()
            if any(
                (distance := _finite_distance(row)) is not None
                and distance <= threshold
                and old_weak_flip(row.get("pred_after"), _parse_int_label(row.get("label")) or 0)
                for row in rows
            )
        }
        output.append({
            "method": method,
            "distance_type": DISTANCE_TYPE,
            "distance_line": DISTANCE_LINE,
            "feature_cost": config.feature_cost,
            "node_mass": config.node_mass,
            "size_penalty_beta": float(config.size_penalty_beta),
            "solver": "exact_emd2",
            "threshold": threshold,
            "threshold_source": threshold_row.get("threshold_source"),
            "quantile": threshold_row.get("quantile"),
            "cf_mode": cf_mode,
            "main_ccrcov_uses": "teacher_strict_flip",
            "teacher_strict_flip_definition": TEACHER_STRICT_FLIP_DEFINITION,
            "old_weak_flip_definition": OLD_WEAK_FLIP_DEFINITION,
            "old_weak_ccrcov_status": "audit_only",
            "num_parents": total_parents,
            "num_teacher_target_parents": len(teacher_target),
            "num_candidates": total_candidates,
            "num_valid_pairs": valid_pairs,
            "num_close_only_covered": len(close_only),
            "close_only_coverage": _rate(len(close_only), total_parents),
            "num_close_cf_covered": len(close_cf),
            "close_cf_coverage": _rate(len(close_cf), total_parents),
            "old_weak_num_close_cf_covered": len(weak),
            "old_weak_close_cf_coverage": _rate(len(weak), total_parents),
            "avg_best_distance": _mean(_finite_distance(row) for row in best_rows),
            "median_best_distance": _median(_finite_distance(row) for row in best_rows),
            "avg_cf_drop_among_covered": _mean(row.get("cf_drop") for row in best_rows),
            "flip_rate_among_covered": _mean(_row_flip(row, label=_parse_int_label(row.get("label")) or 0) for row in best_rows),
            "total_pairs": len(details),
            "cache_hit_rate": cache_hit_rate,
            "node_embedding_cache_hit_rate": node_embedding_cache_hit_rate,
            "skip_redundancy": skip_redundancy,
            **group_audit,
        })
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-csv", default=_env("HIV_CSV", "data/raw/AIDS/HIV.csv"))
    parser.add_argument("--teacher-path", default=_env("TEACHER_PATH"))
    parser.add_argument("--molclr-root", default=_env("MOLCLR_ROOT"))
    parser.add_argument("--molclr-checkpoint", default=_env("MOLCLR_CKPT"))
    parser.add_argument("--label", type=int, default=int(_env("TARGET_LABEL", "1") or 1))
    parser.add_argument("--smiles-col", default=_env("SMILES_COL", "smiles"))
    parser.add_argument("--label-col", default=_env("LABEL_COL", _env("LABEL_COLUMN", "HIV_active")))
    parser.add_argument("--cf-mode", choices=CF_MODES, default=_env("CF_MODE", "strict_flip"))
    parser.add_argument("--output-dir", default=_env("OUTPUT_DIR", "outputs/hpc/eval/ccrcov_molclr_node_wasserstein"))
    parser.add_argument("--max-parents", type=int, default=int(_env("MAX_PARENTS", "200") or 200))
    parser.add_argument("--max-candidates", type=int, default=int(_env("MAX_CANDIDATES", "20") or 20))
    parser.add_argument("--wnode-thresholds", default=_env("WNODE_THRESHOLDS", "auto_quantile"))
    parser.add_argument("--wnode-quantiles", default=_env("WNODE_QUANTILES", DEFAULT_QUANTILES))
    parser.add_argument("--wnode-cache-db", default=_env("WNODE_CACHE_DB", DEFAULT_WNODE_CACHE_DB))
    parser.add_argument("--node-emb-cache-dir", default=_env("NODE_EMB_CACHE_DIR", DEFAULT_NODE_EMB_CACHE_DIR))
    parser.add_argument("--feature-cost", choices=("cosine",), default=_env("FEATURE_COST", "cosine"))
    parser.add_argument("--node-mass", choices=("uniform",), default=_env("NODE_MASS", "uniform"))
    parser.add_argument("--size-penalty-beta", type=float, default=float(_env("SIZE_PENALTY_BETA", "0.0") or 0.0))
    parser.add_argument("--device", default=_env("DEVICE", "cuda"))
    parser.add_argument("--encoder-type", default=_env("MOLCLR_ENCODER_TYPE", "gin"))
    parser.add_argument("--skip-redundancy", type=int, choices=(0, 1), default=1 if _env_bool("SKIP_REDUNDANCY", True) else 0)
    parser.add_argument("--partial-every", type=int, default=int(_env("PARTIAL_EVERY", "500") or 500))
    parser.add_argument("--resume", type=int, choices=(0, 1), default=1 if _env_bool("RESUME", True) else 0)
    parser.add_argument(
        "--run-distance-self-test",
        type=int,
        choices=(0, 1),
        default=1 if _env_bool("RUN_DISTANCE_SELF_TEST", False) else 0,
    )
    parser.add_argument("--run-ours", type=int, choices=(0, 1), default=1 if _env_bool("RUN_OURS", True) else 0)
    parser.add_argument("--run-fullgraph", type=int, choices=(0, 1), default=1 if _env_bool("RUN_FULLGRAPH", False) else 0)
    parser.add_argument("--ours-selected-path", default=_env("OURS_SELECTED_PATH"))
    parser.add_argument("--fullgraph-candidates-path", default=_env("FULLGRAPH_CANDIDATES_PATH", _env("GCF_CANDIDATES_PATH", _env("GT_FULLGRAPH_CANDIDATES_PATH"))))
    parser.add_argument("--fullgraph-method-name", default=_env("FULLGRAPH_METHOD_NAME"))
    parser.add_argument("--selection-method", default=_env("SELECTION_METHOD"))
    parser.add_argument("--preselected-topk", type=int, default=int(_env("PRESELECTED_TOPK", "20") or 20))
    parser.add_argument("--require-preselected-topk", type=int, choices=(0, 1), default=1 if _env_bool("REQUIRE_PRESELECTED_TOPK", True) else 0)
    return parser


def main() -> int:
    started = time.time()
    args = build_parser().parse_args()
    if not args.teacher_path or not args.molclr_root or not args.molclr_checkpoint:
        raise SystemExit("[ERROR] TEACHER_PATH, MOLCLR_ROOT, and MOLCLR_CKPT are required.")
    if args.cf_mode != "strict_flip":
        raise SystemExit("[ERROR] MolCLR-Node-Wasserstein final evaluation requires CF_MODE=strict_flip.")
    if not bool(args.run_ours) and not bool(args.run_fullgraph):
        raise SystemExit("[ERROR] Enable RUN_OURS or RUN_FULLGRAPH.")
    output = ensure_directory(Path(args.output_dir).expanduser())
    details_dir = ensure_directory(output / "details")
    combined_dir = ensure_directory(output / "combined")

    audits: dict[str, dict[str, Any]] = {}
    if bool(args.require_preselected_topk) and int(args.preselected_topk) <= 0:
        raise SystemExit("[ERROR] REQUIRE_PRESELECTED_TOPK=1 requires PRESELECTED_TOPK > 0.")
    try:
        if bool(args.run_ours):
            if not args.ours_selected_path:
                raise ValueError("RUN_OURS=1 requires OURS_SELECTED_PATH")
            audits["ours"] = validate_preselected_ours_directory(args.ours_selected_path, args.preselected_topk)
        if bool(args.run_fullgraph):
            if not args.fullgraph_candidates_path:
                raise ValueError("RUN_FULLGRAPH=1 requires FULLGRAPH_CANDIDATES_PATH")
            audits["fullgraph"] = validate_preselected_candidate_csv(args.fullgraph_candidates_path, args.preselected_topk)
    except ValueError as exc:
        if bool(args.require_preselected_topk):
            raise SystemExit(f"[ERROR] preselected candidate validation failed: {exc}") from exc
        audits = {}

    config = MolCLRNodeWassersteinConfig(
        molclr_root=args.molclr_root,
        molclr_ckpt=args.molclr_checkpoint,
        cache_db=args.wnode_cache_db,
        node_emb_cache_dir=args.node_emb_cache_dir,
        feature_cost=args.feature_cost,
        node_mass=args.node_mass,
        size_penalty_beta=float(args.size_penalty_beta),
        device=args.device,
        encoder_type=args.encoder_type,
    )
    fingerprint_payload = {
        "distance_type": DISTANCE_TYPE,
        "feature_cost": args.feature_cost,
        "node_mass": args.node_mass,
        "size_penalty_beta": args.size_penalty_beta,
        "dataset": _file_identity(args.dataset_csv),
        "teacher": _file_identity(args.teacher_path),
        "molclr_checkpoint": _file_identity(args.molclr_checkpoint),
        "ours_selected": _file_identity(args.ours_selected_path),
        "fullgraph_candidates": _file_identity(args.fullgraph_candidates_path),
        "label": args.label,
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "max_parents": args.max_parents,
        "max_candidates": args.max_candidates,
        "preselected_topk": args.preselected_topk,
    }
    fingerprint = config_fingerprint(fingerprint_payload)
    run_config_base = {
        **fingerprint_payload,
        "config_fingerprint": fingerprint,
        "distance_line": DISTANCE_LINE,
        "distance_type": DISTANCE_TYPE,
        "solver": "exact_emd2",
        "cf_mode": args.cf_mode,
        "main_ccrcov_uses": "teacher_strict_flip",
        "teacher_strict_flip_definition": TEACHER_STRICT_FLIP_DEFINITION,
        "old_weak_flip_definition": OLD_WEAK_FLIP_DEFINITION,
        "old_weak_ccrcov_status": "audit_only",
        "feature_cost": args.feature_cost,
        "node_mass": args.node_mass,
        "size_penalty_beta": float(args.size_penalty_beta),
        "dataset_csv": args.dataset_csv,
        "teacher_path": args.teacher_path,
        "molclr_root": args.molclr_root,
        "molclr_checkpoint": args.molclr_checkpoint,
        "ours_selected_path": args.ours_selected_path,
        "fullgraph_candidates_path": args.fullgraph_candidates_path,
        "run_ours": bool(args.run_ours),
        "run_fullgraph": bool(args.run_fullgraph),
        "candidate_set_preselected": bool(audits),
        "selection_performed_in_eval": False if audits else bool(args.max_candidates),
        "selection_method": args.selection_method or "mixed_external_preselection",
        "preselected_topk": int(args.preselected_topk),
        "require_preselected_topk": bool(args.require_preselected_topk),
        "partial_every": int(args.partial_every),
        "resume": bool(args.resume),
        "skip_redundancy": bool(args.skip_redundancy),
    }
    _write_json(output / "run_config.json", run_config_base)

    print("[MOLCLR_NODE_WASSERSTEIN_CONFIG]", flush=True)
    for key in ("distance_line", "dataset_csv", "teacher_path", "molclr_checkpoint", "feature_cost", "node_mass", "size_penalty_beta", "cf_mode"):
        print(f"{key}={run_config_base[key]}", flush=True)

    provider = MolCLRNodeWassersteinDistance(config)
    if bool(args.run_distance_self_test):
        identical = provider.distance("CCO", "CCO")
        forward = provider.distance("CCO", "CCN")
        reverse = provider.distance("CCN", "CCO")
        values = [identical.get("distance"), forward.get("distance"), reverse.get("distance")]
        if not all(result.get("ok") and value is not None and math.isfinite(float(value)) for result, value in zip((identical, forward, reverse), values)):
            raise SystemExit("[ERROR] MolCLR-Node-Wasserstein distance self-test returned an invalid result.")
        if float(values[0]) > 1e-7:
            raise SystemExit(f"[ERROR] d(CCO,CCO)={values[0]} exceeds 1e-7.")
        if abs(float(values[1]) - float(values[2])) > 1e-9:
            raise SystemExit("[ERROR] MolCLR-Node-Wasserstein symmetry self-test failed.")
        _write_json(output / "distance_self_test.json", {
            "d_CCO_CCO": float(values[0]),
            "d_CCO_CCN": float(values[1]),
            "d_CCN_CCO": float(values[2]),
            "identical_tolerance": 1e-7,
            "symmetry_tolerance": 1e-9,
            "pass": True,
        })
    _dataset_path, parents, _actual_label_col = _load_parent_records(
        args.dataset_csv,
        label=int(args.label),
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        max_parents=int(args.max_parents) if int(args.max_parents) > 0 else None,
    )
    teacher = TeacherSemanticScorer(args.teacher_path)
    groups: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []

    def evaluate_group(kind: str, method: str, candidates: list[Any], audit: dict[str, Any] | None) -> None:
        checkpoint = ResumeCheckpoint(output, kind, fingerprint, bool(args.resume))
        try:
            initial, completed = checkpoint.load()
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
        common = dict(
            parents=parents,
            candidates=candidates,
            teacher=teacher,
            provider=provider,
            distance_type=DISTANCE_TYPE,
            distance_name=DISTANCE_NAME,
            partial_path=checkpoint.partial_path,
            partial_every=int(args.partial_every),
            initial_details=initial,
            completed_pair_keys=completed,
            checkpoint_callback=checkpoint.save,
            write_builtin_partial=False,
        )
        if kind == "ours":
            details = _evaluate_ours(**common)
            row_unit = "match_instance"
        else:
            details = _evaluate_gt_fullgraph(method=method, **common)
            row_unit = "parent_candidate_pair"
        for row in details:
            row.update({
                "distance_line": DISTANCE_LINE,
                "feature_cost": args.feature_cost,
                "node_mass": args.node_mass,
                "size_penalty_beta": float(args.size_penalty_beta),
                "solver": "exact_emd2",
                "skip_redundancy": bool(args.skip_redundancy),
            })
        row_audit = build_evaluation_row_audit(details, evaluation_row_unit=row_unit)
        selection_method = args.selection_method or str((audit or {}).get("selection_method") or "external_preselected")
        group_audit = {
            "candidate_set_preselected": audit is not None,
            "selection_performed_in_eval": False if audit is not None else bool(args.max_candidates),
            "selection_method": selection_method,
            **row_audit,
        }
        groups.append({"method": method, "details": details, "num_candidates": len(candidates), **group_audit})
        all_details.extend(details)

    if bool(args.run_ours):
        _path, candidates = _load_candidate_records(
            args.ours_selected_path, fields=OURS_FRAGMENT_FIELDS, directory_candidates=OURS_DIRECTORY_CANDIDATES
        )
        if "ours" not in audits and int(args.max_candidates) > 0:
            candidates = candidates[: int(args.max_candidates)]
        if "ours" in audits:
            expected = [_fragment_identity(value) for value in audits["ours"].get("ordered_fragments", [])]
            if [_fragment_identity(candidate.smiles) for candidate in candidates] != expected:
                raise SystemExit("[ERROR] Ours loader changed external selector order.")
        evaluate_group("ours", "ours_selected_subgraphs", candidates, audits.get("ours"))

    if bool(args.run_fullgraph):
        _path, candidates = _load_candidate_records(
            args.fullgraph_candidates_path, fields=GT_FULLGRAPH_FIELDS, directory_candidates=GT_DIRECTORY_CANDIDATES
        )
        if "fullgraph" not in audits and int(args.max_candidates) > 0:
            candidates = candidates[: int(args.max_candidates)]
        method = infer_fullgraph_method(args.fullgraph_candidates_path, args.fullgraph_method_name)
        evaluate_group("fullgraph", method, candidates, audits.get("fullgraph"))

    if not groups:
        raise SystemExit("[ERROR] No evaluation groups were produced.")
    if str(args.wnode_thresholds).strip() in {"", "auto", "auto_quantile"}:
        ours_details = next((group["details"] for group in groups if group["method"] == "ours_selected_subgraphs"), None)
        if ours_details is None:
            raise SystemExit("[ERROR] auto_quantile WNode thresholds may only be calibrated from Ours.")
        threshold_reference = ours_details
    else:
        threshold_reference = all_details
    threshold_rows, thresholds = build_threshold_table(args.wnode_thresholds, args.wnode_quantiles, threshold_reference)
    _write_csv(output / "distance_quantiles.csv", threshold_rows, ["threshold_source", "quantile", "threshold"])

    stats = provider.stats_dict()
    summaries: list[dict[str, Any]] = []
    for group in groups:
        group_audit = {key: group[key] for key in (
            "candidate_set_preselected", "selection_performed_in_eval", "selection_method",
            "evaluation_row_unit", "num_unique_parent_candidate_pairs", "num_detail_rows", "num_valid_match_instances"
        )}
        summaries.extend(summarize_method(
            method=group["method"], details=group["details"], threshold_rows=threshold_rows,
            total_parents=len(parents), total_candidates=group["num_candidates"], config=config,
            cf_mode=args.cf_mode,
            cache_hit_rate=float(stats.get("pair_distance_cache_hit_rate") or 0.0),
            node_embedding_cache_hit_rate=float(stats.get("node_embedding_cache_hit_rate") or 0.0),
            skip_redundancy=bool(args.skip_redundancy), group_audit=group_audit,
        ))
    _write_csv(details_dir / "pair_details.csv", all_details, _detail_fields(all_details))
    _write_csv(combined_dir / "combined_threshold_summary.csv", summaries, SUMMARY_FIELDS)
    _write_json(combined_dir / "combined_threshold_summary.json", {"threshold_summary": summaries})

    methods = {str(group["selection_method"]) for group in groups}
    run_config = {
        **run_config_base,
        "thresholds": thresholds,
        "threshold_source": "explicit" if str(args.wnode_thresholds).strip() not in {"", "auto", "auto_quantile"} else "auto_quantile",
        "wnode_quantiles": _parse_float_list(args.wnode_quantiles),
        "candidate_set_preselected": all(bool(group["candidate_set_preselected"]) for group in groups),
        "selection_performed_in_eval": any(bool(group["selection_performed_in_eval"]) for group in groups),
        "selection_method": next(iter(methods)) if len(methods) == 1 else "mixed",
        "method_evaluation_audits": {
            str(group["method"]): {key: group[key] for key in (
                "candidate_set_preselected", "selection_performed_in_eval", "selection_method",
                "evaluation_row_unit", "num_unique_parent_candidate_pairs", "num_detail_rows", "num_valid_match_instances"
            )} for group in groups
        },
    }
    _write_json(output / "run_config.json", run_config)
    stats["runtime_seconds"] = float(time.time() - started)
    _write_json(output / "cache_stats.json", stats)
    _write_json(output / "_RUN_COMPLETE.json", {
        "complete": True,
        "config_fingerprint": fingerprint,
        "num_detail_rows": len(all_details),
        "num_unique_parent_candidate_pairs": sum(group["num_unique_parent_candidate_pairs"] for group in groups),
        "completed_at_epoch_seconds": time.time(),
    })
    provider.close()
    print(f"[TASK_DONE] output_dir={output} runtime_seconds={stats['runtime_seconds']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
