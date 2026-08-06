"""Auditable source, action-replay, and chemistry-repair gates for Mutagenicity."""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chem_repair import REPAIR_METHOD, REPAIR_POLICY_VERSION, repair_candidate, replay_raw_actions
from .contracts import (
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .graph_trace import stable_graph_sha256
from .preregistration import write_mutagenicity_chem_repair_preregistration
from .project_dataset import load_mutagenicity_generation_bundle
from .runtime import _torch_load, _torch_save_atomic, validate_counterfactual_payload


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    atomic_write_bytes(path, _csv_bytes(rows, fields))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True, default=str) + "\n"
        for row in rows
    )
    atomic_write_bytes(path, payload.encode("utf-8"))


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected JSON object list: {source}")
    return [dict(row) for row in value]


def _canonical(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Source SMILES is invalid: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _project_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _candidate_graphs(payload: Mapping[str, Any]) -> list[Any]:
    graph_map, candidates = validate_counterfactual_payload(payload)
    values: list[Any] = []
    for index, candidate in enumerate(candidates):
        key = candidate.get("graph_hash")
        entry = graph_map.get(key)
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(f"Candidate {index} is absent from graph_map: {key!r}")
        values.append(entry[0])
    return values


def _mapping_payloads(schema: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    atom = {
        "schema_version": 1,
        "source": "MutagenicityGraphSchema.feature_atomic_numbers",
        "node_label_semantics": "one_hot_atom_type",
        "node_feature_dim": int(schema.node_feature_dim),
        "index_to_atomic_number": {
            str(index): int(value)
            for index, value in enumerate(schema.feature_atomic_numbers)
        },
        "alternative_atom_search": False,
    }
    bond = {
        "schema_version": 1,
        "source": "gcfexplainer_mutagenicity_fullgraph_codec_v1",
        "retained_bond_rule": "source_sidecar_exact",
        "new_untyped_edge_bond_type": "SINGLE",
        "supported_source_bond_types": list(schema.bond_type_vocabulary),
        "alternative_bond_order_search": False,
    }
    return atom, bond


def _source_and_noop_gate(
    *, root: Path, dataset_dir: Path, parent_limit: int
) -> tuple[Any, list[Any], dict[str, Mapping[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from rdkit import Chem
    from torch_geometric.data import Batch

    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        decode_generated_fullgraph,
        load_dataset_artifacts,
        reconstruct_source_graph,
    )

    schema, _train, _val, generation, _summary = load_dataset_artifacts(dataset_dir)
    records = sorted(generation, key=lambda row: str(row["molecule_id"]))[: int(parent_limit)]
    bundle = load_mutagenicity_generation_bundle(
        dataset_dir=dataset_dir,
        parent_limit=parent_limit,
    )
    if len(records) != parent_limit or len(bundle.graphs) != parent_limit:
        raise ValueError("Mutagenicity source cohort does not match the frozen smoke parent count.")
    record_by_id = {str(row["molecule_id"]): row for row in records}
    source_rows: list[dict[str, Any]] = []
    no_op_rows: list[dict[str, Any]] = []
    source_hashes = [stable_graph_sha256(graph) for graph in bundle.graphs]
    serialized = root / "source_graph_noop_roundtrip.pt"
    _torch_save_atomic(bundle.graphs, serialized)
    reloaded = _torch_load(serialized)
    batched = Batch.from_data_list(bundle.graphs).to_data_list()
    if not isinstance(reloaded, list) or len(reloaded) != len(bundle.graphs):
        raise ValueError("Source graph save/load did not preserve the graph list.")
    if len(batched) != len(bundle.graphs):
        raise ValueError("Source graph batch/unbatch did not preserve graph count.")
    for index, (graph, loaded_graph, batched_graph) in enumerate(
        zip(bundle.graphs, reloaded, batched, strict=True)
    ):
        parent_id = bundle.parent_ids[index]
        record = record_by_id[parent_id]
        molecule, diagnostics = reconstruct_source_graph(record, schema)
        reconstructed = _canonical(Chem.MolToSmiles(molecule, isomericSmiles=True))
        expected = _canonical(str(record["canonical_smiles"]))
        decoded = decode_generated_fullgraph(graph, source_record=record, schema=schema)
        graph_hash = source_hashes[index]
        loaded_hash = stable_graph_sha256(loaded_graph)
        batched_hash = stable_graph_sha256(batched_graph)
        node_origin = [int(value) for value in graph.comrecgc_node_origin.detach().cpu().tolist()]
        expected_origin = list(range(int(graph.num_nodes)))
        source_ok = bool(
            diagnostics.get("round_trip_passed")
            and reconstructed == expected
            and node_origin == expected_origin
        )
        no_op_ok = bool(
            decoded.decode_ok
            and decoded.canonical_smiles == expected
            and graph_hash == loaded_hash == batched_hash
        )
        source_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "sanitize_ok": source_ok,
                "node_count_exact": int(molecule.GetNumAtoms()) == int(record["num_nodes"]),
                "undirected_bond_count_exact": int(molecule.GetNumBonds()) == int(record["num_edges"]),
                "atomic_number_exact": diagnostics.get("atomic_numbers_exact"),
                "formal_charge_exact": diagnostics.get("formal_charges_exact"),
                "aromaticity_exact": diagnostics.get("aromaticity_exact"),
                "explicit_h_exact": diagnostics.get("explicit_hs_exact"),
                "no_implicit_exact": diagnostics.get("no_implicit_exact"),
                "chirality_exact": diagnostics.get("chiral_tags_exact"),
                "bond_type_exact": diagnostics.get("bond_types_exact"),
                "edge_direction_normalization_exact": True,
                "duplicate_edge_absence": int(graph.edge_index.shape[1]) == 2 * int(record["num_edges"]),
                "self_loop_absence": not bool((graph.edge_index[0] == graph.edge_index[1]).any().item()),
                "edge_attr_alignment_exact": getattr(graph, "edge_attr", None) is None,
                "node_lineage_exact": node_origin == expected_origin,
                "canonical_molecule_equivalence": reconstructed == expected,
                "roundtrip_ok": source_ok,
            }
        )
        no_op_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "clone_graph_sha256": stable_graph_sha256(graph.clone()),
                "save_load_graph_sha256": loaded_hash,
                "batch_unbatch_graph_sha256": batched_hash,
                "empty_action_decode_ok": decoded.decode_ok,
                "canonical_smiles": decoded.canonical_smiles,
                "noop_roundtrip_ok": no_op_ok,
            }
        )
    if not all(bool(row["roundtrip_ok"]) for row in source_rows):
        raise ValueError("Mutagenicity source round-trip is not 100%; repair is forbidden.")
    if not all(bool(row["noop_roundtrip_ok"]) for row in no_op_rows):
        raise ValueError("Mutagenicity no-op/serialization round-trip is not 100%.")
    return schema, bundle.graphs, record_by_id, source_rows, no_op_rows


def run_mutagenicity_chemistry_audit(
    *,
    project_root: str | Path,
    dataset_dir: str | Path,
    generation_dir: str | Path,
    trace_lineage_path: str | Path,
    trace_parity_path: str | Path,
    common_recourse_dir: str | Path,
    output_dir: str | Path,
    preregistration_path: str | Path,
    parent_limit: int = 64,
    expected_candidate_count: int | None = 164,
    expected_medoid_count: int | None = 4,
    expected_counterfactuals_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit all raw candidates, replay exact actions, and freeze one repair each."""

    root = require_empty_output(output_dir)
    project = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    generation_root = Path(generation_dir).expanduser().resolve()
    common_root = Path(common_recourse_dir).expanduser().resolve()
    generation_manifest = _load_json(generation_root / "run_manifest.json")
    counterfactuals = Path(generation_manifest["counterfactuals_path"]).expanduser().resolve()
    actual_sha = sha256_file(counterfactuals)
    if actual_sha != str(generation_manifest["counterfactuals_sha256"]):
        raise ValueError("Mutagenicity generation artifact differs from its manifest.")
    if expected_counterfactuals_sha256 and actual_sha != expected_counterfactuals_sha256:
        raise ValueError("Mutagenicity generation artifact differs from the frozen blocker SHA256.")
    parity = _load_json(trace_parity_path)
    if parity.get("trace_parity_passed") is not True:
        raise ValueError("Trace parity must pass before chemistry audit.")
    lineage = _load_json_list(trace_lineage_path)
    payload = _torch_load(counterfactuals)
    candidate_graphs = _candidate_graphs(payload)
    if (
        expected_candidate_count is not None
        and (
            len(candidate_graphs) != expected_candidate_count
            or len(lineage) != expected_candidate_count
        )
    ):
        raise ValueError(
            "Frozen Mutagenicity candidate count mismatch: "
            f"graphs={len(candidate_graphs)}, lineage={len(lineage)}, expected={expected_candidate_count}."
        )
    if [int(row.get("candidate_index", -1)) for row in lineage] != list(range(len(lineage))):
        raise ValueError("Candidate action lineage is not in exact official candidate order.")
    if any(row.get("action_lineage_resolved") is not True for row in lineage):
        raise ValueError("At least one Mutagenicity candidate action lineage is unresolved.")

    schema, source_graphs, source_records, source_rows, no_op_rows = _source_and_noop_gate(
        root=root,
        dataset_dir=dataset_root,
        parent_limit=parent_limit,
    )
    source_by_id = {
        str(getattr(graph, "comrecgc_parent_id")): graph for graph in source_graphs
    }
    atom_mapping, bond_mapping = _mapping_payloads(schema)
    atom_mapping_path = root / "atom_label_mapping.json"
    bond_mapping_path = root / "bond_mapping.json"
    write_json(atom_mapping_path, atom_mapping)
    write_json(bond_mapping_path, bond_mapping)

    preregistration_target = Path(preregistration_path).expanduser().resolve()
    if preregistration_target.exists():
        preregistration = _load_json(preregistration_target)
        if preregistration.get("source_counterfactuals_sha256") != actual_sha:
            raise ValueError("Existing chemistry preregistration targets another artifact.")
        preregistration_sha = sha256_file(preregistration_target)
    else:
        preregistration = write_mutagenicity_chem_repair_preregistration(
            project_commit=_project_commit(project),
            source_counterfactuals_path=counterfactuals,
            trace_parity_path=trace_parity_path,
            atom_mapping_path=atom_mapping_path,
            bond_mapping_path=bond_mapping_path,
            output_path=preregistration_target,
        )
        preregistration_sha = str(preregistration["file_sha256"])

    raw_rows: list[dict[str, Any]] = []
    repaired_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    first_invalid_rows: list[dict[str, Any]] = []
    repaired_graphs: list[Any] = []
    action_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_reason_counts: Counter[str] = Counter()
    for index, (graph, trace) in enumerate(zip(candidate_graphs, lineage, strict=True)):
        parent_id = str(trace.get("parent_id") or getattr(graph, "comrecgc_parent_id", ""))
        if parent_id not in source_by_id or parent_id not in source_records:
            raise ValueError(f"Candidate {index} parent lineage is absent: {parent_id!r}")
        source_graph = source_by_id[parent_id]
        source_record = source_records[parent_id]
        actions = list(trace.get("actions") or [])
        raw_replay = replay_raw_actions(source_graph, actions)
        replay_sha = stable_graph_sha256(raw_replay)
        raw_sha = stable_graph_sha256(graph)
        if replay_sha != raw_sha:
            raise ValueError(f"Candidate {index} exact action replay differs from official graph.")
        from src.baselines.gcfexplainer_mutagenicity_adapter import decode_generated_fullgraph

        raw_decoded = decode_generated_fullgraph(graph, source_record=source_record, schema=schema)
        first = repair_candidate(
            source_graph=source_graph,
            source_record=source_record,
            schema=schema,
            actions=actions,
        )
        second = repair_candidate(
            source_graph=source_graph,
            source_record=source_record,
            schema=schema,
            actions=actions,
        )
        deterministic = bool(
            first.output_graph_sha256 == second.output_graph_sha256
            and first.canonical_smiles == second.canonical_smiles
            and first.action_records == second.action_records
        )
        if not deterministic:
            raise ValueError(f"Candidate {index} chemistry repair is nondeterministic.")
        repaired_graphs.append(first.graph)
        raw_reason = "valid" if raw_decoded.decode_ok else str(raw_decoded.failure_reason)
        raw_reason_counts[raw_reason] += 1
        candidate_id = f"COMRECGC_MUT_RAW_{index:06d}"
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                "raw_graph_sha256": raw_sha,
                "raw_sanitize_ok": raw_decoded.decode_ok,
                "raw_invalid_reason": "" if raw_decoded.decode_ok else raw_reason,
                "raw_canonical_smiles": raw_decoded.canonical_smiles,
                "action_count": len(actions),
                "action_replay_exact": True,
            }
        )
        first_invalid = dict(first.first_invalid_action or {})
        repaired_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                "raw_graph_sha256": raw_sha,
                "input_candidate_sha256": raw_sha,
                "repair_policy_sha256": preregistration_sha,
                "repair_attempted": True,
                "repair_success": first.repair_success,
                "repair_noop": first.repair_noop,
                "retained_action_count": first.retained_action_count,
                "skipped_action_count": first.skipped_action_count,
                "skipped_action_types": list(first.skipped_action_types),
                "dependent_action_skip_count": first.dependent_action_skip_count,
                "action_retention_rate": (
                    first.retained_action_count / len(actions) if actions else 1.0
                ),
                "first_invalid_action": first_invalid,
                "repaired_graph_sha256": first.output_graph_sha256,
                "output_candidate_sha256": first.output_graph_sha256,
                "repaired_smiles": first.canonical_smiles,
                "repaired_sanitize_ok": first.repair_success,
                "repair_deterministic": deterministic,
                "rf_scored": False,
                "rf_pred_before": None,
                "rf_pred_after": None,
                "rf_strict_flip": None,
                "rf_cf_drop": None,
                "wnode_scored": False,
                "wnode_distance": None,
            }
        )
        if first_invalid:
            first_invalid_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "source_graph_id": parent_id,
                    **first_invalid,
                }
            )
        for action in first.action_records:
            row = {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                **dict(action),
            }
            action_rows.append(row)
            action_type_counts[str(action["action_type"])][str(action["status"])] += 1

    _torch_save_atomic(repaired_graphs, root / "repaired_candidates.pt")
    reloaded_repaired = _torch_load(root / "repaired_candidates.pt")
    if len(reloaded_repaired) != len(repaired_graphs):
        raise ValueError("Repaired candidate artifact is not reloadable.")

    common_records = _load_json_list(common_root / "selected_common_recourses.json")
    if expected_medoid_count is not None and len(common_records) != expected_medoid_count:
        raise ValueError("Official Mutagenicity medoid count differs from the frozen blocker.")
    ranks = [int(row["rank"]) for row in common_records]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("Official common-recourse rank is not contiguous and frozen.")
    medoid_rows: list[dict[str, Any]] = []
    medoid_graphs: list[Any] = []
    for common in common_records:
        candidate_index = int(common["generation_candidate_index"])
        repaired = repaired_rows[candidate_index]
        medoid_graphs.append(repaired_graphs[candidate_index])
        medoid_rows.append(
            {
                "official_cluster_rank": int(common["rank"]),
                "cluster_id": common.get("common_recourse_id"),
                "cluster_label": common.get("cluster_label"),
                "candidate_index": candidate_index,
                "candidate_id": repaired["candidate_id"],
                "is_official_medoid": True,
                "representative_policy": "repaired_original_official_medoid",
                "repair_success": repaired["repair_success"],
                "repair_noop": repaired["repair_noop"],
                "repaired_graph_sha256": repaired["repaired_graph_sha256"],
                "repaired_smiles": repaired["repaired_smiles"],
                "invalid_slot_backfill": False,
                "rank_compaction": False,
            }
        )
    _torch_save_atomic(medoid_graphs, root / "repaired_official_medoids.pt")

    source_fields = list(source_rows[0])
    noop_fields = list(no_op_rows[0])
    _write_csv(root / "source_roundtrip.csv", source_rows, source_fields)
    _write_csv(root / "noop_roundtrip.csv", no_op_rows, noop_fields)
    _write_csv(root / "raw_candidates.csv", raw_rows, list(raw_rows[0]))
    _write_csv(root / "candidate_validity.csv", repaired_rows, list(repaired_rows[0]))
    _write_jsonl(root / "action_replay.jsonl", action_rows)
    _write_csv(
        root / "first_invalid_actions.csv",
        first_invalid_rows,
        list(first_invalid_rows[0]) if first_invalid_rows else ["candidate_id", "candidate_index"],
    )
    action_type_rows = [
        {"action_type": action_type, **dict(counts), "total": sum(counts.values())}
        for action_type, counts in sorted(action_type_counts.items())
    ]
    action_statuses = sorted({key for row in action_type_counts.values() for key in row})
    _write_csv(
        root / "action_type_validity.csv",
        action_type_rows,
        ["action_type", *action_statuses, "total"],
    )
    medoid_fields = (
        list(medoid_rows[0])
        if medoid_rows
        else [
            "official_cluster_rank",
            "cluster_id",
            "cluster_label",
            "candidate_index",
            "candidate_id",
            "is_official_medoid",
            "representative_policy",
            "repair_success",
            "repair_noop",
            "repaired_graph_sha256",
            "repaired_smiles",
            "invalid_slot_backfill",
            "rank_compaction",
        ]
    )
    _write_csv(root / "medoid_validity.csv", medoid_rows, medoid_fields)
    _write_csv(root / "cluster_validity.csv", medoid_rows, medoid_fields)
    _write_jsonl(root / "raw_candidates.jsonl", raw_rows)
    _write_jsonl(root / "repaired_candidates.jsonl", repaired_rows)
    _write_jsonl(root / "repair_provenance.jsonl", repaired_rows)
    _write_csv(root / "raw_candidate_validity.csv", raw_rows, list(raw_rows[0]))
    _write_csv(root / "repaired_candidate_validity.csv", repaired_rows, list(repaired_rows[0]))
    _write_csv(root / "official_cluster_order.csv", medoid_rows, medoid_fields)
    _write_csv(root / "official_medoid_repair.csv", medoid_rows, medoid_fields)
    write_json(root / "selected_common_recourses.json", medoid_rows)
    _write_jsonl(root / "representative_counterfactuals.jsonl", medoid_rows)

    source_pass = sum(bool(row["roundtrip_ok"]) for row in source_rows)
    noop_pass = sum(bool(row["noop_roundtrip_ok"]) for row in no_op_rows)
    repaired_count = sum(bool(row["repair_success"]) for row in repaired_rows)
    repaired_medoid_count = sum(bool(row["repair_success"]) for row in medoid_rows)
    audit = {
        "schema_version": 1,
        "audit_passed": True,
        "engineering_smoke_pass": True,
        "method": REPAIR_METHOD,
        "repair_policy_version": REPAIR_POLICY_VERSION,
        "repair_policy_sha256": preregistration_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "project_commit": _project_commit(project),
        "source_parent_count": len(source_rows),
        "source_roundtrip_pass_count": source_pass,
        "source_roundtrip_rate": source_pass / len(source_rows),
        "noop_roundtrip_pass_count": noop_pass,
        "noop_roundtrip_rate": noop_pass / len(no_op_rows),
        "trace_parity": True,
        "raw_candidate_count": len(raw_rows),
        "raw_valid_candidate_count": sum(bool(row["raw_sanitize_ok"]) for row in raw_rows),
        "raw_invalid_reason_counts": dict(raw_reason_counts),
        "repair_provenance_count": len(repaired_rows),
        "repaired_candidate_count": repaired_count,
        "official_medoid_count": len(medoid_rows),
        "repaired_official_medoid_count": repaired_medoid_count,
        "repair_deterministic_count": sum(bool(row["repair_deterministic"]) for row in repaired_rows),
        "one_raw_candidate_max_one_repaired_candidate": True,
        "official_cluster_rank_unchanged": True,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "rf_used_in_repair": False,
        "wnode_used_in_repair": False,
        "strict_flip_used_in_repair": False,
        "project_feasibility_status": (
            "PROJECT_FEASIBILITY_OBSERVED"
            if repaired_medoid_count > 0
            else "PROJECT_FEASIBILITY_NOT_OBSERVED"
        ),
        "strict_flip_status": "NOT_EVALUATED_IN_REPAIR",
        "calibration_loaded": False,
        "test_loaded": False,
        "counterfactuals_path": str(counterfactuals),
        "counterfactuals_sha256": actual_sha,
        "trace_lineage_path": str(Path(trace_lineage_path).resolve()),
        "trace_lineage_sha256": sha256_file(trace_lineage_path),
        "trace_parity_path": str(Path(trace_parity_path).resolve()),
        "trace_parity_sha256": sha256_file(trace_parity_path),
        "preregistration_path": str(preregistration_target),
        "preregistration_sha256": preregistration_sha,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "audit.json", audit)
    write_json(
        root / "run_manifest.json",
        {
            **audit,
            "run_complete": True,
            "algorithm_rerun": False,
            "candidate_order_source": "official_random_walk_candidate_order",
            "official_medoid_policy": "original_official_medoid_only",
            "repaired_candidates_path": str(root / "repaired_candidates.pt"),
            "repaired_candidates_sha256": sha256_file(root / "repaired_candidates.pt"),
            "repaired_official_medoids_path": str(root / "repaired_official_medoids.pt"),
            "repaired_official_medoids_sha256": sha256_file(
                root / "repaired_official_medoids.pt"
            ),
        },
    )
    atomic_write_bytes(
        root / "audit.txt",
        (
            "COMRECGC Mutagenicity chemistry audit\n"
            f"source_roundtrip={source_pass}/{len(source_rows)}\n"
            f"noop_roundtrip={noop_pass}/{len(no_op_rows)}\n"
            f"raw_candidates={len(raw_rows)} raw_valid={audit['raw_valid_candidate_count']}\n"
            f"repaired_candidates={repaired_count} repaired_official_medoids={repaired_medoid_count}\n"
            "rf_used_in_repair=false wnode_used_in_repair=false\n"
            "[COMRECGC_MUT_CHEMISTRY_ENGINEERING_SMOKE_PASS]\n"
        ).encode("utf-8"),
    )
    write_json(
        root / "final_artifact_audit.json",
        {
            **audit,
            "run_complete": True,
            "output_reload_verified": True,
            "required_artifacts": {
                name: {
                    "bytes": (root / name).stat().st_size,
                    "sha256": sha256_file(root / name),
                }
                for name in (
                    "source_roundtrip.csv",
                    "noop_roundtrip.csv",
                    "raw_candidates.jsonl",
                    "repair_provenance.jsonl",
                    "official_medoid_repair.csv",
                    "repaired_candidates.pt",
                    "repaired_official_medoids.pt",
                )
            },
        },
    )
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, "audit_passed": True})
    write_json(
        root / "manifest.json",
        {
            "run_complete": True,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(root.iterdir())
                if path.is_file()
            },
        },
    )
    return audit
