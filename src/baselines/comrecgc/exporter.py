"""Decode COMRECGC graph medoids and freeze RF strict-flip candidates."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    ordered_ids_sha256,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("COMRECGC candidate export requires torch.") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, default=str))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_frozen_candidate_manifest(
    *,
    dataset: str,
    selected: Sequence[Mapping[str, Any]],
    csv_path: Path,
) -> dict[str, Any]:
    candidate_ids = [str(row["candidate_id"]) for row in selected]
    return {
        "schema_version": "comrecgc_frozen_top20_v1",
        "method": METHOD,
        "dataset": dataset,
        "candidate_count": len(selected),
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "selection_method": "official_comrecgc_greedy_cluster_order_filtered_by_validity",
        "adaptation_mode": ADAPTATION_MODE,
        "cf_mode": CF_MODE,
        "distance_line": DISTANCE_LINE,
        "selected_candidate_ids": candidate_ids,
        "selected_candidate_order_sha256": ordered_ids_sha256(candidate_ids),
        "file_inventory": {
            csv_path.name: {
                "bytes": csv_path.stat().st_size,
                "sha256": sha256_file(csv_path),
            }
        },
        "calibration_loaded": False,
        "test_loaded": False,
    }


def export_gate_failure(
    summary: Mapping[str, Any], *, require_top_k: bool, top_k: int
) -> tuple[str, str] | None:
    if int(summary.get("rf_scored_count", 0)) < 1:
        return (
            "NoRFScoredCandidates",
            "COMRECGC chemical codec smoke gate requires rf_scored_count >= 1.",
        )
    if require_top_k and int(summary.get("selected_count", 0)) < int(top_k):
        return (
            "InsufficientStrictFlipCandidates",
            "COMRECGC did not produce the required frozen strict-flip candidate count.",
        )
    return None


def _aids_schema_and_record(graph: Any, atom_vocabulary: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    from rdkit import Chem
    from src.baselines.gcfexplainer_mutagenicity_adapter import MutagenicityGraphSchema

    source_smiles = str(getattr(graph, "comrecgc_source_smiles", ""))
    source_id = str(getattr(graph, "comrecgc_parent_id", ""))
    molecule = Chem.MolFromSmiles(source_smiles)
    if molecule is None:
        raise ValueError("aids_source_smiles_parse_failed")
    Chem.SanitizeMol(molecule)
    periodic = Chem.GetPeriodicTable()
    feature_atomic_numbers = tuple(
        int(periodic.GetAtomicNumber(str(symbol))) for symbol in atom_vocabulary
    )
    if len(feature_atomic_numbers) != int(graph.x.shape[1]):
        raise ValueError("aids_atom_vocabulary_dimension_mismatch")
    atoms = [
        {
            "graph_node_index": int(atom.GetIdx()),
            "original_atom_index": int(atom.GetIdx()),
            "attached_original_atom_index": None,
            "atomic_num": int(atom.GetAtomicNum()),
            "formal_charge": int(atom.GetFormalCharge()),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "num_explicit_hs": int(atom.GetNumExplicitHs()),
            "no_implicit": bool(atom.GetNoImplicit()),
            "chiral_tag": int(atom.GetChiralTag()),
            "isotope": int(atom.GetIsotope()),
        }
        for atom in molecule.GetAtoms()
    ]
    bonds: list[dict[str, Any]] = []
    for bond in molecule.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        bonds.append(
            {
                "begin": min(begin, end),
                "end": max(begin, end),
                "rdkit_begin": begin,
                "rdkit_end": end,
                "source_bond_index": int(bond.GetIdx()),
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "is_conjugated": bool(bond.GetIsConjugated()),
                "stereo": int(bond.GetStereo()),
                "bond_dir": int(bond.GetBondDir()),
                "stereo_atoms": [int(value) for value in bond.GetStereoAtoms()],
            }
        )
    schema = MutagenicityGraphSchema(
        atom_vocabulary=feature_atomic_numbers,
        feature_atomic_numbers=feature_atomic_numbers,
        formal_charge_vocabulary=tuple(sorted({int(atom["formal_charge"]) for atom in atoms})),
        aromaticity_vocabulary=tuple(sorted({bool(atom["is_aromatic"]) for atom in atoms})),
        bond_type_vocabulary=tuple(sorted({str(bond["bond_type"]) for bond in bonds})),
        max_num_nodes=max(len(atoms), int(graph.num_nodes)),
        explicit_h_nodes=False,
    )
    return schema, {
        "molecule_id": source_id,
        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "atom_sidecar": atoms,
        "bond_sidecar": bonds,
    }


def _sync_generated_node_lineage(graph: Any) -> None:
    """Expose COMRECGC's updated node lineage to the shared fullgraph codec."""

    lineage = getattr(graph, "comrecgc_node_origin", None)
    if lineage is None:
        raise ValueError("generated_missing_source_lineage")
    # Mutagenicity source graphs already carry gcf_node_origin.  Official edit
    # operations clone that field unchanged, while our neighbor wrapper updates
    # comrecgc_node_origin for node additions/removals.  The latter is therefore
    # the authoritative lineage for a generated graph.
    graph.gcf_node_origin = lineage


def decode_representative(
    graph: Any,
    *,
    dataset: str,
    atom_vocabulary: Sequence[str | int],
) -> dict[str, Any]:
    from src.baselines.gcfexplainer_mutagenicity_adapter import decode_generated_fullgraph

    try:
        if dataset == "mutagenicity":
            source_record = getattr(graph, "comrecgc_source_record")
            from src.baselines.gcfexplainer_mutagenicity_adapter import MutagenicityGraphSchema

            feature_atoms = tuple(int(value) for value in atom_vocabulary)
            schema = MutagenicityGraphSchema(
                atom_vocabulary=feature_atoms,
                feature_atomic_numbers=feature_atoms,
                formal_charge_vocabulary=(-1, 0, 1),
                aromaticity_vocabulary=(False, True),
                bond_type_vocabulary=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
                max_num_nodes=max(int(graph.num_nodes), int(source_record["num_nodes"])),
            )
        else:
            schema, source_record = _aids_schema_and_record(
                graph, [str(value) for value in atom_vocabulary]
            )
        _sync_generated_node_lineage(graph)
        result = decode_generated_fullgraph(graph, source_record=source_record, schema=schema)
        return {
            "decode_ok": bool(result.decode_ok),
            "canonical_smiles": result.canonical_smiles,
            "raw_smiles": result.raw_smiles,
            "decode_reason": result.failure_reason,
            "projected_new_edge_count": int(result.projected_new_edge_count),
            "retained_edge_count": int(result.retained_edge_count),
            "removed_source_edge_count": int(result.removed_source_edge_count),
            "inherited_atom_state_count": int(result.inherited_atom_state_count),
            "reset_atom_state_count": int(result.reset_atom_state_count),
            "source_parent_id": result.source_parent_id,
            "source_smiles": str(getattr(graph, "comrecgc_source_smiles", "")),
        }
    except Exception as exc:
        return {
            "decode_ok": False,
            "canonical_smiles": "",
            "raw_smiles": "",
            "decode_reason": str(exc) or type(exc).__name__,
            "projected_new_edge_count": 0,
            "retained_edge_count": 0,
            "removed_source_edge_count": 0,
            "inherited_atom_state_count": 0,
            "reset_atom_state_count": 0,
            "source_parent_id": str(getattr(graph, "comrecgc_parent_id", "")),
            "source_smiles": str(getattr(graph, "comrecgc_source_smiles", "")),
        }


def export_representatives(
    *,
    dataset: str,
    common_recourse_dir: str | Path,
    teacher_path: str | Path,
    atom_vocabulary: Sequence[str | int],
    output_dir: str | Path,
    top_k: int = 20,
    require_top_k: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    from src.rewards.teacher_semantic import TeacherSemanticScorer

    root = require_empty_output(output_dir, resume=resume)
    recourse_root = Path(common_recourse_dir).expanduser().resolve()
    manifest = json.loads((recourse_root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_complete") is not True:
        raise ValueError("Common-recourse run is incomplete.")
    rows = json.loads((recourse_root / "selected_common_recourses.json").read_text(encoding="utf-8"))
    graphs = list(_torch_load(recourse_root / "representative_counterfactuals.pt"))
    if len(rows) != len(graphs) or not rows:
        raise ValueError("Representative graph and lineage records are not aligned.")
    teacher = TeacherSemanticScorer(teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(f"Unified RF teacher unavailable: {teacher.availability_reason}")
    audit_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    reasons: Counter[str] = Counter()
    for native_row, graph in zip(rows, graphs, strict=True):
        rank = int(native_row["rank"])
        decoded = decode_representative(
            graph,
            dataset=dataset,
            atom_vocabulary=atom_vocabulary,
        )
        source_score: dict[str, Any] = {}
        target_score: dict[str, Any] = {}
        rejection_stage = "selected"
        rejection_reason = "selected"
        rf_attempted = False
        rf_ok = False
        rf_flip = False
        if not decoded["decode_ok"]:
            rejection_stage = "graph_decode"
            rejection_reason = decoded["decode_reason"]
        elif decoded["canonical_smiles"] in canonical_seen:
            rejection_stage = "canonical_dedup"
            rejection_reason = "duplicate_canonical_smiles"
        else:
            rf_attempted = True
            source_score = teacher.score_smiles(decoded["source_smiles"], label=1)
            target_score = teacher.score_smiles(decoded["canonical_smiles"], label=0)
            rf_ok = bool(source_score.get("teacher_result_ok")) and bool(
                target_score.get("teacher_result_ok")
            )
            source_label = source_score.get("teacher_label")
            target_label = target_score.get("teacher_label")
            rf_flip = bool(rf_ok and int(source_label) == 1 and int(target_label) == 0)
            if not rf_ok:
                rejection_stage = "rf_inference"
                rejection_reason = "rf_query_failed"
            elif not rf_flip:
                rejection_stage = "rf_target_filter"
                rejection_reason = f"strict_flip_failed:{source_label}->{target_label}"
            elif len(selected) >= int(top_k):
                rejection_stage = "after_top_k"
                rejection_reason = "valid_after_frozen_budget"
            else:
                canonical_seen.add(decoded["canonical_smiles"])
                selected.append(
                    {
                        "rank": len(selected) + 1,
                        "native_rank": rank,
                        "candidate_id": str(native_row["candidate_id"]),
                        "smiles": decoded["canonical_smiles"],
                        "canonical_smiles": decoded["canonical_smiles"],
                        "source_parent_id": decoded["source_parent_id"],
                        "source_smiles": decoded["source_smiles"],
                        "rf_pred_before": int(source_score["teacher_label"]),
                        "rf_pred_after": int(target_score["teacher_label"]),
                        "rf_prob_0": float(target_score["teacher_prob"]),
                        "rf_prob_1": 1.0 - float(target_score["teacher_prob"]),
                        "rf_cf_flip": True,
                        "source_method": METHOD,
                        "selection_method": "official_comrecgc_greedy_cluster_order_filtered_by_validity",
                        "adaptation_mode": ADAPTATION_MODE,
                        "candidate_set_preselected": True,
                        "selection_performed_in_eval": False,
                        "calibration_loaded": False,
                        "test_loaded": False,
                        "projected_new_edge_count": decoded["projected_new_edge_count"],
                        "retained_edge_count": decoded["retained_edge_count"],
                    }
                )
        reasons[rejection_reason] += 1
        audit_rows.append(
            {
                "candidate_id": str(native_row["candidate_id"]),
                "native_rank": rank,
                **decoded,
                "rf_inference_attempted": rf_attempted,
                "rf_inference_ok": rf_ok,
                "rf_prediction_before": source_score.get("teacher_label"),
                "rf_prediction_after": target_score.get("teacher_label"),
                "rf_cf_flip": rf_flip,
                "selected": bool(selected and selected[-1]["native_rank"] == rank),
                "rejection_stage": rejection_stage,
                "rejection_reason": rejection_reason,
            }
        )
    _write_jsonl(root / "candidate_filter_audit.jsonl", audit_rows)
    fields = [
        "rank",
        "native_rank",
        "candidate_id",
        "smiles",
        "canonical_smiles",
        "source_parent_id",
        "source_smiles",
        "rf_pred_before",
        "rf_pred_after",
        "rf_prob_0",
        "rf_prob_1",
        "rf_cf_flip",
        "source_method",
        "selection_method",
        "adaptation_mode",
        "candidate_set_preselected",
        "selection_performed_in_eval",
        "calibration_loaded",
        "test_loaded",
        "projected_new_edge_count",
        "retained_edge_count",
    ]
    _write_csv(root / "selected_top20.csv", selected, fields)
    selected_order_hash = ordered_ids_sha256([row["candidate_id"] for row in selected])
    summary = {
        "method": METHOD,
        "dataset": dataset,
        "distance_line": DISTANCE_LINE,
        "cf_mode": CF_MODE,
        "adaptation_mode": ADAPTATION_MODE,
        "native_recourse_count": len(rows),
        "audit_row_count": len(audit_rows),
        "decode_ok_count": sum(bool(row["decode_ok"]) for row in audit_rows),
        "rf_scored_count": sum(bool(row["rf_inference_ok"]) for row in audit_rows),
        "rf_strict_flip_count": sum(bool(row["rf_cf_flip"]) for row in audit_rows),
        "selected_count": len(selected),
        "requested_top_k": int(top_k),
        "available_k": len(selected),
        "rejection_reason_counts": dict(reasons),
        "candidate_order_source": "official_greedy_common_recourse_order",
        "selected_candidate_order_sha256": selected_order_hash,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "teacher_path": str(Path(teacher_path).expanduser().resolve()),
        "teacher_sha256": sha256_file(teacher_path),
        "common_recourse_manifest_path": str(recourse_root / "run_manifest.json"),
        "common_recourse_manifest_sha256": sha256_file(recourse_root / "run_manifest.json"),
        "selected_top20_path": str(root / "selected_top20.csv"),
        "selected_top20_sha256": sha256_file(root / "selected_top20.csv"),
        "run_complete": len(selected) >= int(top_k),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "run_manifest.json", summary)
    frozen_manifest = build_frozen_candidate_manifest(
        dataset=dataset,
        selected=selected,
        csv_path=root / "selected_top20.csv",
    )
    write_json(root / "frozen_candidate_manifest.json", frozen_manifest)
    gate_failure = export_gate_failure(
        summary, require_top_k=require_top_k, top_k=int(top_k)
    )
    if gate_failure is not None:
        error_class, message = gate_failure
        failure = {
            **summary,
            "stage": "project_candidate_export",
            "error_class": error_class,
            "message": message,
            "run_complete": False,
            "controlled_retry_allowed": error_class
            in {"NoRFScoredCandidates", "InsufficientStrictFlipCandidates"},
        }
        (root / "_SMOKE_AUDIT_COMPLETE.json").unlink(missing_ok=True)
        (root / "_RUN_COMPLETE.json").unlink(missing_ok=True)
        write_json(root / "failure_summary.json", failure)
        write_json(root / "_RUN_FAILED.json", failure)
        raise RuntimeError(f"{error_class}: {message}")
    (root / "_RUN_FAILED.json").unlink(missing_ok=True)
    (root / "failure_summary.json").unlink(missing_ok=True)
    marker = "_RUN_COMPLETE.json" if len(selected) >= int(top_k) else "_SMOKE_AUDIT_COMPLETE.json"
    write_json(root / marker, {"run_complete": True, "available_k": len(selected)})
    return summary
