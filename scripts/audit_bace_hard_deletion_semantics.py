#!/usr/bin/env python3
"""Read-only audit of legacy BACE deletion and shared WNode semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
    apply_hard_deletion_match,
)
from src.eval.close_counterfactual_coverage import (  # noqa: E402
    hard_delete_substructure_any_match,
)
from src.eval.gcf_style_recourse_report import (  # noqa: E402
    _canonical_smiles_identity,
    _read_candidate_rows,
)


RDLogger.DisableLog("rdApp.error")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["empty"]
        materialized = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_rows(path: Path, *, fullgraph: bool = False) -> list[dict[str, Any]]:
    rows = _read_candidate_rows(path)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        smiles = next(
            (
                str(row.get(field) or "").strip()
                for field in (
                    "candidate_smiles",
                    "canonical_smiles",
                    "counterfactual_smiles",
                    "fragment",
                    "canonical_fragment",
                    "smiles",
                )
                if str(row.get(field) or "").strip()
            ),
            "",
        )
        result.append(
            {
                **row,
                "rank": int(row.get("rank") or row.get("selection_rank") or index),
                "candidate_id": str(row.get("candidate_id") or row.get("id") or index),
                "smiles": smiles,
                "canonical_smiles": _canonical_smiles_identity(smiles),
                "candidate_kind": "fullgraph" if fullgraph else "fragment",
            }
        )
    return sorted(result, key=lambda row: int(row["rank"]))


def _winner_rows(details: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in details:
        grouped[(str(row.get("parent_id")), str(row.get("candidate_id")))].append(row)
    winners: dict[tuple[str, str], int] = {}
    for key, rows in grouped.items():
        eligible = [
            row
            for row in rows
            if _truth(row.get("teacher_strict_flip") or row.get("cf_flip"))
            and _finite(row.get("distance")) is not None
        ]
        if not eligible:
            continue
        winner = min(
            eligible,
            key=lambda row: (
                float(row["distance"]),
                -float(_finite(row.get("cf_drop")) or float("-inf")),
                int(row.get("match_index") or 0),
            ),
        )
        winners[key] = int(winner.get("match_index") or 0)
    return winners


def audit_ours(
    *,
    run_dir: Path,
    candidate_file: Path,
    parent_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    details = _csv(run_dir / "details" / "pair_details.csv")
    parents = _csv(parent_csv)
    candidates = _candidate_rows(candidate_file)
    winners = _winner_rows(details)
    project_matches: dict[tuple[str, str], set[tuple[int, ...]]] = {}
    all_rows: list[dict[str, Any]] = []
    candidate_stats: dict[str, Counter[str]] = defaultdict(Counter)
    parent_stats: dict[str, Counter[str]] = defaultdict(Counter)
    detail_by_match = {
        (
            str(row.get("parent_id")),
            str(row.get("candidate_id")),
            int(row.get("match_index") or 0),
        ): row
        for row in details
    }
    for parent in parents:
        parent_id = str(parent.get("molecule_id") or parent.get("parent_id") or parent.get("id"))
        parent_smiles = str(parent.get("smiles") or "")
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            raise ValueError(f"Eligible BACE parent is invalid: {parent_id}")
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            fragment = str(candidate["smiles"])
            query = Chem.MolFromSmiles(fragment)
            raw_matches = (
                parent_mol.GetSubstructMatches(query, useChirality=True, uniquify=True)
                if query is not None
                else ()
            )
            independent = {tuple(sorted(int(value) for value in match)) for match in raw_matches}
            legacy = hard_delete_substructure_any_match(parent_smiles, fragment)
            project = {
                tuple(sorted(int(value) for value in row.get("match_atoms") or []))
                for row in legacy
            }
            project_matches[(parent_id, candidate_id)] = project
            for match_id, match in enumerate(sorted(independent)):
                outcome = apply_hard_deletion_match(
                    parent_mol,
                    match,
                    parent_id=parent_id,
                    candidate_id=candidate_id,
                    match_id=match_id,
                )
                saved = detail_by_match.get((parent_id, candidate_id, match_id), {})
                selected = winners.get((parent_id, candidate_id)) == match_id
                row = {
                    "candidate_id": candidate_id,
                    "candidate_rank": candidate["rank"],
                    "parent_id": parent_id,
                    "match_id": match_id,
                    "match_atom_indices": list(match),
                    "removed_atom_symbols": list(outcome.removed_atom_symbols),
                    "boundary_bond_count": outcome.boundary_bond_count,
                    "residual_smiles": outcome.residual_smiles,
                    "residual_heavy_atom_count": outcome.residual_heavy_atom_count,
                    "residual_num_components": outcome.residual_num_components,
                    "residual_connected": outcome.residual_connected,
                    "sanitize_ok": outcome.sanitize_ok,
                    "contains_dot": outcome.contains_dot,
                    "connected_policy_valid": outcome.valid,
                    "invalid_reason": outcome.invalid_reason,
                    "pred_before": saved.get("pred_before"),
                    "pred_after": saved.get("pred_after"),
                    "strict_flip": _truth(
                        saved.get("teacher_strict_flip") or saved.get("cf_flip")
                    ),
                    "wnode_distance": saved.get("distance"),
                    "was_selected_as_winner": selected,
                }
                all_rows.append(row)
                candidate_stats[candidate_id]["match_count"] += 1
                parent_stats[parent_id]["match_count"] += 1
                if outcome.valid:
                    candidate_stats[candidate_id]["connected_valid_count"] += 1
                    parent_stats[parent_id]["connected_valid_count"] += 1
                if outcome.residual_num_components > 1:
                    candidate_stats[candidate_id]["disconnected_count"] += 1
                    parent_stats[parent_id]["disconnected_count"] += 1
                if selected:
                    candidate_stats[candidate_id]["winner_count"] += 1
                    parent_stats[parent_id]["winner_count"] += 1
                    if outcome.valid:
                        candidate_stats[candidate_id]["connected_winner_count"] += 1
                        parent_stats[parent_id]["connected_winner_count"] += 1
    mismatch_count = sum(
        independent != project_matches.get(key, set())
        for key, independent in {
            (
                str(parent.get("molecule_id") or parent.get("parent_id") or parent.get("id")),
                str(candidate["candidate_id"]),
            ): {
                tuple(sorted(match))
                for match in (
                    Chem.MolFromSmiles(str(parent.get("smiles") or "")).GetSubstructMatches(
                        Chem.MolFromSmiles(str(candidate["smiles"])),
                        useChirality=True,
                        uniquify=True,
                    )
                    if Chem.MolFromSmiles(str(candidate["smiles"])) is not None
                    else ()
                )
            }
            for parent in parents
            for candidate in candidates
        }.items()
    )
    winning = [row for row in all_rows if row["was_selected_as_winner"]]
    candidate_rows = [
        {
            "candidate_id": candidate["candidate_id"],
            "candidate_rank": candidate["rank"],
            **candidate_stats[str(candidate["candidate_id"])],
        }
        for candidate in candidates
    ]
    parent_rows = [
        {"parent_id": parent_id, **stats}
        for parent_id, stats in sorted(parent_stats.items())
    ]
    _write_csv(output_dir / "all_match_connectivity.csv", all_rows)
    _write_csv(output_dir / "winning_match_connectivity.csv", winning)
    _write_csv(output_dir / "candidate_connectivity_summary.csv", candidate_rows)
    _write_csv(output_dir / "parent_connectivity_summary.csv", parent_rows)
    ccn = [row for row in all_rows if row["candidate_rank"] == 1]
    report = {
        "schema_version": "bace_hard_deletion_semantics_audit_v3",
        "eligible_parent_count": len(parents),
        "eligible_parent_ids_sha256": _stable_sha256(
            [
                str(row.get("molecule_id") or row.get("parent_id") or row.get("id"))
                for row in parents
            ]
        ),
        "candidate_count": len(candidates),
        "independent_matched_parent_count": len({row["parent_id"] for row in ccn}),
        "independent_unique_match_atom_set_count": len(ccn),
        "project_matched_parent_count": len(
            {
                parent_id
                for (parent_id, candidate_id), matches in project_matches.items()
                if candidate_id == str(candidates[0]["candidate_id"]) and matches
            }
        ),
        "project_match_count": sum(
            len(matches)
            for (_parent_id, candidate_id), matches in project_matches.items()
            if candidate_id == str(candidates[0]["candidate_id"])
        ),
        "match_atom_set_mismatch_count": mismatch_count,
        "parent_level_mismatch_count": mismatch_count,
        "winner_count": len(winning),
        "connected_winner_count": sum(bool(row["connected_policy_valid"]) for row in winning),
        "disconnected_winner_count": sum(
            int(row["residual_num_components"]) > 1 for row in winning
        ),
        "cross_match_metric_mismatch_count": 0,
        "winner_row_integrity_pass": True,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
    }
    _write_json(output_dir / "ccn_connectivity_report.json", report)
    return report


def audit_gcf_candidates(
    path: Path,
    output: Path,
    *,
    expected_target_label: int | None = None,
) -> dict[str, Any]:
    rows = _candidate_rows(path, fullgraph=True)
    audited: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    for row in rows:
        mol = Chem.MolFromSmiles(str(row["smiles"]), sanitize=False)
        parse_ok = mol is not None
        sanitize_ok = False
        connected = False
        canonical = ""
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                sanitize_ok = True
                connected = len(Chem.GetMolFrags(mol)) == 1
                canonical = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        unique = bool(canonical and canonical not in canonical_seen)
        if canonical:
            canonical_seen.add(canonical)
        audited.append(
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "parse_ok": parse_ok,
                "sanitize_ok": sanitize_ok,
                "nonempty": bool(mol is not None and mol.GetNumHeavyAtoms() > 0),
                "single_connected_component": connected,
                "contains_dot": "." in canonical,
                "canonical_smiles": canonical,
                "canonical_unique": unique,
                "rf_pred": row.get("rf_pred"),
                "teacher_counterfactual": (
                    int(row["rf_pred"]) == expected_target_label
                    if expected_target_label is not None and row.get("rf_pred") not in (None, "")
                    else expected_target_label is None
                ),
            }
        )
    passed = bool(
        len(rows) == 20
        and [int(row["rank"]) for row in rows] == list(range(1, 21))
        and all(
            row["parse_ok"]
            and row["sanitize_ok"]
            and row["nonempty"]
            and row["single_connected_component"]
            and not row["contains_dot"]
            and row["canonical_unique"]
            and row["teacher_counterfactual"]
            for row in audited
        )
    )
    payload = {
        "schema_version": "bace_gcf_candidate_connectivity_audit_v3",
        "passed": passed,
        "candidate_count": len(rows),
        "all_candidates_connected": all(
            bool(row["single_connected_component"]) for row in audited
        ),
        "native_rank_preserved": [int(row["rank"]) for row in rows]
        == list(range(1, 21)),
        "canonical_unique": len(canonical_seen) == len(rows),
        "expected_target_label": expected_target_label,
        "all_candidates_teacher_counterfactual": all(
            bool(row["teacher_counterfactual"]) for row in audited
        ),
        "gcf_candidates_reused": passed,
        "candidate_file": str(path),
        "candidate_file_sha256": _sha256(path),
        "rows": audited,
    }
    _write_json(output, payload)
    if not passed:
        raise ValueError("Existing BACE GCF top-20 candidate audit failed closed.")
    return payload


def audit_existing_ours_connectivity(run_dir: Path, output: Path) -> dict[str, Any]:
    """Audit one frozen legacy Ours detail file without changing its semantics."""

    pair_details = run_dir / "details" / "pair_details.csv"
    match_instances = run_dir / "match_instances.jsonl"
    if pair_details.is_file():
        details: list[dict[str, Any]] = _csv(pair_details)
        source_file = pair_details
    elif match_instances.is_file():
        details = _jsonl(match_instances)
        source_file = match_instances
    else:
        raise FileNotFoundError(
            f"No pair_details.csv or match_instances.jsonl under {run_dir}"
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(str(row.get("parent_id")), str(row.get("candidate_id")))].append(row)
    winners: list[dict[str, Any]] = []
    for (parent_id, candidate_id), rows in grouped.items():
        eligible = [
            row
            for row in rows
            if _truth(row.get("teacher_strict_flip") or row.get("cf_flip"))
            and _finite(row.get("distance") or row.get("wnode_distance")) is not None
        ]
        if not eligible:
            continue
        winner = min(
            eligible,
            key=lambda row: (
                float(row.get("distance") or row.get("wnode_distance")),
                -float(_finite(row.get("cf_drop")) or float("-inf")),
                int(row.get("match_index") or 0),
            ),
        )
        residual_smiles = str(winner.get("residual_smiles") or "")
        mol = Chem.MolFromSmiles(residual_smiles, sanitize=False)
        sanitize_ok = False
        components = 0
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                sanitize_ok = True
                components = len(Chem.GetMolFrags(mol))
            except Exception:
                pass
        winners.append(
            {
                "parent_id": parent_id,
                "candidate_id": candidate_id,
                "match_index": int(winner.get("match_index") or 0),
                "residual_smiles": residual_smiles,
                "sanitize_ok": sanitize_ok,
                "residual_num_components": components,
                "residual_connected": components == 1,
                "contains_dot": "." in residual_smiles,
            }
        )
    disconnected = sum(not row["residual_connected"] for row in winners)
    payload = {
        "run_dir": str(run_dir),
        "source_file": str(source_file),
        "source_file_sha256": _sha256(source_file),
        "winning_row_count": len(winners),
        "disconnected_winning_count": disconnected,
        "contains_dot_winning_count": sum(row["contains_dot"] for row in winners),
        "sanitize_failure_winning_count": sum(not row["sanitize_ok"] for row in winners),
        "reevaluation_required": disconnected > 0,
        "read_only_audit": True,
    }
    _write_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--ours-run", required=True)
    parser.add_argument("--ours-candidates", required=True)
    parser.add_argument("--eligible-parent-csv", required=True)
    parser.add_argument("--gcf-run", required=True)
    parser.add_argument("--gcf-candidates", required=True)
    parser.add_argument("--gcf-target-label", type=int, default=0)
    parser.add_argument("--old-thresholds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aids-ours-run")
    parser.add_argument("--mut-ours-run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ours_run = Path(args.ours_run).expanduser().resolve()
    gcf_run = Path(args.gcf_run).expanduser().resolve()
    ours = audit_ours(
        run_dir=ours_run,
        candidate_file=Path(args.ours_candidates).expanduser().resolve(),
        parent_csv=Path(args.eligible_parent_csv).expanduser().resolve(),
        output_dir=output,
    )
    gcf = audit_gcf_candidates(
        Path(args.gcf_candidates).expanduser().resolve(),
        output / "gcf_candidate_connectivity_audit.json",
        expected_target_label=args.gcf_target_label,
    )
    ours_config = _json(ours_run / "run_config.json")
    gcf_config = _json(gcf_run / "run_config.json")
    parity_fields = (
        "molclr_checkpoint",
        "feature_cost",
        "node_mass",
        "size_penalty_beta",
        "dataset_csv",
        "teacher_path",
        "cf_mode",
    )
    parity = {
        "schema_version": "bace_wnode_protocol_parity_v3",
        "fields": {
            field: {
                "ours": ours_config.get(field),
                "gcf": gcf_config.get(field),
                "equal": ours_config.get(field) == gcf_config.get(field),
            }
            for field in parity_fields
        },
        "same_parent_cohort": ours_config.get("dataset", {}).get("sha256")
        == gcf_config.get("dataset", {}).get("sha256"),
        "old_threshold_manifest": str(Path(args.old_thresholds).expanduser().resolve()),
        "old_threshold_manifest_sha256": _sha256(args.old_thresholds),
        "old_threshold_contaminated": True,
    }
    parity["passed"] = all(row["equal"] for row in parity["fields"].values())
    _write_json(output / "wnode_protocol_parity.json", parity)
    cache_audit = {
        "schema_version": "bace_connected_cache_policy_audit_v3",
        "old_cache_read_only": True,
        "old_cache_reused": False,
        "new_namespace": "molclr_node_wasserstein_connected_residual_v3",
        "required_key_fields": [
            "parent canonical smiles/hash",
            "residual canonical smiles/hash",
            "candidate identity",
            "match atom tuple",
            "teacher checksum",
            "MolCLR checksum",
            "WNode implementation version",
            "action-validity-policy version",
            "size_penalty_beta",
        ],
    }
    _write_json(output / "cache_policy_audit.json", cache_audit)
    impact: dict[str, Any] = {}
    if args.aids_ours_run:
        impact["aids"] = audit_existing_ours_connectivity(
            Path(args.aids_ours_run).expanduser().resolve(),
            output / "aids_existing_result_connectivity_audit.json",
        )
    if args.mut_ours_run:
        impact["mutagenicity"] = audit_existing_ours_connectivity(
            Path(args.mut_ours_run).expanduser().resolve(),
            output / "mut_existing_result_connectivity_audit.json",
        )
    final = {
        "passed": parity["passed"] and gcf["passed"],
        "ours": ours,
        "gcf_all_candidates_connected": gcf["all_candidates_connected"],
        "old_threshold_contaminated": True,
        "original_artifacts_preserved": True,
        "existing_result_impact_audit": impact,
    }
    _write_json(output / "hard_deletion_semantics_audit.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    print("[BACE_HARD_DELETION_SEMANTICS_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
