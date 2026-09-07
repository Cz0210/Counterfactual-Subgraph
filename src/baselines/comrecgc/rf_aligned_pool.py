"""AIDS-only frozen native-pool replay and RF-aligned candidate screening.

No search, classifier fitting, OT, pair-store or DBSCAN is performed here.
The old recorded actions and candidate hashes own identity; missing evidence
is a provenance gap, never an invented action or a chemical repair.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

RF_SHA = "df33f46b0c4474b3d8ff0ffb8bbc08483c9b497170aab82590ff4505428d0c71"
POLICY = "global_first_recorded_exact_event_in_selected_trace_order_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_events(input_root: Path, *, verify_hashes: bool = True):
    manifest = json.loads((input_root / "trace/selected_action_trace_manifest.json").read_text())
    chunks = manifest["chunks"]
    for index, chunk in enumerate(chunks):
        path = input_root / "trace/selected_action_trace_chunks" / Path(chunk["path"]).name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular trace chunk {path}")
        if verify_hashes and file_sha(path) != chunk["sha256"]:
            raise ValueError(f"Trace chunk hash mismatch: {index}")
        count = 0
        with path.open() as handle:
            for line in handle:
                count += 1
                yield json.loads(line)
        if count != int(chunk["row_count"]):
            raise ValueError(f"Trace chunk row count mismatch: {index}")


def predecessor_index(events) -> tuple[dict[str, dict[str, Any]], int]:
    index: dict[str, dict[str, Any]] = {}
    count = 0
    for raw in events:
        if raw.get("event") != "selected_transition":
            continue
        count += 1
        if raw.get("action_resolution") != "exact" or not raw.get("action"):
            raise ValueError("No exact recorded selected action; inference is prohibited")
        index.setdefault(str(raw["target_official_hash"]), dict(raw))
    return index, count


def recorded_path(row: Mapping[str, Any], predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    cursor = str(row["official_graph_hash"])
    seen: set[str] = set()
    reverse = []
    while cursor in predecessor:
        if cursor in seen:
            raise ValueError("Selected-action predecessor cycle")
        seen.add(cursor)
        event = predecessor[cursor]
        if str(event["parent_id"]) != str(row["parent_id"]):
            raise ValueError("Recorded predecessor crosses authoritative parent")
        reverse.append(event)
        cursor = str(event["source_official_hash"])
    result = list(reversed(reverse))
    if len(result) != int(row["action_count"]):
        raise ValueError("Recorded action count differs from sealed lineage index")
    if not row.get("action_lineage_resolved"):
        raise ValueError("Original candidate lineage was not resolved")
    return result


def replay_candidate(row, predecessor, parents):
    from .chem_repair import apply_action_to_graph
    from .graph_trace import stable_untyped_graph_sha256, trace_node_ids
    path = recorded_path(row, predecessor)
    graph = parents[str(row["parent_id"])].clone()
    graph.edge_attr = None
    for position, event in enumerate(path):
        actual = stable_untyped_graph_sha256(graph)
        if actual != event["source_graph_sha256"]:
            raise ValueError(f"Recorded source SHA mismatch at action {position}: {actual}")
        node_ids = trace_node_ids(graph)
        if str(event["action"][0]) in {"NA", "INA"}:
            node_ids.append(f"new:{row['parent_id']}:move:{int(event['move_index'])}:head:{int(event['head_index'])}:path:{position}:target:{event['target_graph_sha256']}")
        elif str(event["action"][0]) in {"NR", "INR"}:
            node_ids.pop(int(event["action"][1]))
        graph = apply_action_to_graph(graph, event["action"], target_node_ids=node_ids)
        actual = stable_untyped_graph_sha256(graph)
        if actual != event["target_graph_sha256"]:
            raise ValueError(f"Recorded target SHA mismatch at action {position}: {actual}")
    if stable_untyped_graph_sha256(graph) != row["stable_graph_sha256"]:
        raise ValueError("Replayed candidate differs from sealed candidate graph")
    return graph


def compact_graph(graph) -> dict[str, Any]:
    """Lossless one-hot native graph container, not a SMILES re-encoding."""
    import torch
    labels = graph.x.argmax(dim=1)
    onehot = torch.nn.functional.one_hot(labels, graph.x.shape[1]).to(graph.x.dtype)
    if not torch.equal(graph.x, onehot):
        raise ValueError("Native features are not exact one-hot; compact conversion refused")
    return {"labels": labels.tolist(), "feature_count": graph.x.shape[1],
            "x_dtype": str(graph.x.dtype), "edge_index": graph.edge_index.tolist(),
            "node_origin": graph.comrecgc_node_origin.tolist(),
            "source_parent_id": graph.comrecgc_parent_id,
            "source_smiles": graph.comrecgc_source_smiles}


def rf_predict(smiles: list[str], bundle) -> list[dict[str, Any]]:
    import numpy as np
    from src.rewards.reward_calculator import smiles_to_morgan_array
    model = bundle["model"]
    classes = [int(item) for item in model.classes_]
    if sorted(classes) != [0, 1]:
        raise ValueError(f"Unexpected RF class mapping: {classes}")
    fingerprints = [smiles_to_morgan_array(s, radius=int(bundle["fingerprint_radius"]),
                     n_bits=int(bundle["fingerprint_bits"]), clean_dummy_atoms=False) for s in smiles]
    if any(value is None for value in fingerprints):
        raise ValueError("RF input fingerprint missing for chemically accepted graph")
    probabilities = model.predict_proba(np.stack(fingerprints))
    if not np.isfinite(probabilities).all():
        raise ValueError("RF returned non-finite probabilities")
    return [{"prediction": classes[int(np.argmax(prob))],
             "p0": float(prob[classes.index(0)]), "p1": float(prob[classes.index(1)])}
            for prob in probabilities]


def screen_pool(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    import torch
    from rdkit import Chem, RDLogger
    from .project_dataset import load_aids_generation_bundle
    from .exporter import decode_representative
    from src.rewards.reward_calculator import load_oracle_bundle
    RDLogger.DisableLog("rdApp.error")
    torch.set_num_threads(int(config.get("threads", 2)))
    if config["rf_sha256"] != RF_SHA:
        raise ValueError("RF contract is not the authorized AIDS frozen oracle")
    output_root.mkdir(parents=True, exist_ok=True)
    contract_sha = digest(config)
    contract_path = output_root / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != dict(config):
        raise ValueError("Existing pool screening contract differs")
    if (output_root / "terminal.json").exists():
        return json.loads((output_root / "terminal.json").read_text())
    atomic_json(contract_path, dict(config))
    start = time.monotonic()
    source = load_aids_generation_bundle(dataset_dir=config["dataset_dir"], source_csv=config["source_csv"])
    if len(source.graphs) != 1283:
        raise ValueError(f"AIDS source scope must retain 1283, got {len(source.graphs)}")
    parents = dict(zip(source.parent_ids, source.graphs, strict=True))
    atomic_json(output_root / "source_input_binding.json", source.audit())
    rf = load_oracle_bundle(config["rf_path"])
    if hasattr(rf["model"], "n_jobs"):
        rf["model"].n_jobs = int(config.get("threads", 2))
    source_scores = rf_predict([g.comrecgc_source_smiles for g in source.graphs], rf)
    atomic_json(output_root / "source_predictions.json", {"scope": "AIDS_HIV_EXISTING_1283_GENERATION_OVERLAP_NOT_UNSEEN_TEST", "classes": [int(x) for x in rf["model"].classes_], "rows": [dict(parent_id=pid, **score) for pid, score in zip(source.parent_ids, source_scores, strict=True)], "source1_count": sum(x["prediction"] == 1 for x in source_scores), "denominator": 1283})
    input_root = Path(config["input_root"])
    lineage_file = input_root / "trace/candidate_action_lineage_index.jsonl"
    lineage_sha = file_sha(lineage_file)
    if lineage_sha != config["candidate_lineage_sha256"]:
        raise ValueError("Candidate lineage index content binding failed")
    predecessor, event_count = predecessor_index(checked_events(input_root))
    segments = output_root / "segments"
    segments.mkdir(exist_ok=True)
    done = 0
    counts: Counter[str] = Counter()
    rf_cache: dict[str, dict[str, Any]] = {}
    for path in sorted(segments.glob("segment-*.json")):
        segment = json.loads(path.read_text())
        if segment["start"] != done or segment["contract_sha"] != contract_sha:
            raise ValueError("Screening segment checkpoint continuity failed")
        for row in segment["rows"]:
            counts[row["state"]] += 1
            if "rf" in row:
                rf_cache[row["canonical_smiles"]] = row["rf"]
        done += len(segment["rows"])
    original_done = done
    pending: list[dict[str, Any]] = []
    chunk_size = int(config.get("checkpoint_candidates", 500))

    def commit() -> None:
        nonlocal done, pending
        if not pending:
            return
        unique = list(dict.fromkeys(row["canonical_smiles"] for row in pending if row["state"] == "CHEM_VALID" and row["canonical_smiles"] not in rf_cache))
        for offset in range(0, len(unique), 256):
            batch = unique[offset:offset + 256]
            rf_cache.update(zip(batch, rf_predict(batch, rf), strict=True))
        for row in pending:
            if row["state"] == "CHEM_VALID":
                row["rf"] = rf_cache[row["canonical_smiles"]]
                row["state"] = "RF_TARGET0" if row["rf"]["prediction"] == 0 else "RF_TARGET_REJECT"
                if row["state"] != "RF_TARGET0":
                    row.pop("graph", None)
            counts[row["state"]] += 1
        atomic_json(segments / f"segment-{done:09d}.json", {"contract_sha": contract_sha, "start": done, "rows": pending})
        done += len(pending)
        pending = []
        atomic_json(output_root / "progress.json", {"state": "RUNNING", "completed_candidates": done, "expected_candidates": config["expected_candidates"], "counts": dict(counts), "rf_unique_chemical_graphs": len(rf_cache), "elapsed_seconds": time.monotonic() - start, "pid": os.getpid()})

    with lineage_file.open() as handle:
        for index, line in enumerate(handle):
            if index < done:
                continue
            original = json.loads(line)
            if original["candidate_index"] != index:
                raise ValueError("Candidate index order mismatch")
            row = {key: original[key] for key in ("candidate_index", "official_graph_hash", "stable_graph_sha256", "parent_id", "action_count")}
            try:
                graph = replay_candidate(original, predecessor, parents)
            except (ValueError, IndexError, KeyError) as exc:
                row.update(state="CACHE_PROVENANCE_GAP", reason=str(exc))
            else:
                decoded = decode_representative(graph, dataset="aids", atom_vocabulary=source.atom_vocabulary)
                row.update(decode=decoded)
                smiles = decoded["canonical_smiles"]
                molecule = Chem.MolFromSmiles(smiles) if decoded["decode_ok"] else None
                if molecule is None or len(Chem.GetMolFrags(molecule)) != 1 or any(a.GetAtomicNum() == 0 for a in molecule.GetAtoms()):
                    row.update(state="CHEM_REJECT", reason=decoded["decode_reason"] or "disconnected_or_dummy")
                else:
                    row.update(state="CHEM_VALID", canonical_smiles=Chem.MolToSmiles(molecule, canonical=True), graph=compact_graph(graph))
            pending.append(row)
            if len(pending) >= chunk_size:
                commit()
    commit()
    if done != int(config["expected_candidates"]):
        raise ValueError(f"Incomplete frozen pool: {done} != {config['expected_candidates']}")
    unique_target0 = sum(score["prediction"] == 0 for score in rf_cache.values())
    result = {"state": "POOL_SCREEN_COMPLETE" if not counts["CACHE_PROVENANCE_GAP"] else "EVIDENCE_INSUFFICIENT", "completed_candidates": done, "counts": dict(counts), "unique_chemical_graphs": len(rf_cache), "unique_rf_target0": unique_target0, "trace_events": event_count, "policy": POLICY, "source_denominator": 1283, "source1_count": sum(x["prediction"] == 1 for x in source_scores), "generation_oracle": "HISTORICAL_GNN", "acceptance_oracle": "FROZEN_AIDS_RF", "variant": "GNN_PROPOSED_RF_VALIDATED_ADAPTATION", "generation_rerun": False, "test_loaded": False, "pair_store_created": False, "elapsed_seconds": time.monotonic() - start, "resumed_candidates": original_done, "contract_sha": contract_sha}
    atomic_json(output_root / "terminal.json", result)
    return result


def repair_screen_gaps(config: Mapping[str, Any], *, source_root: Path, output_root: Path):
    """Reconcile only explicit replay gaps; preserve all successful source rows."""
    import torch
    from rdkit import Chem, RDLogger
    from .project_dataset import load_aids_generation_bundle
    from .exporter import decode_representative
    from src.rewards.reward_calculator import load_oracle_bundle
    terminal = json.loads((source_root / "terminal.json").read_text())
    if terminal["state"] not in {"POOL_SCREEN_COMPLETE", "EVIDENCE_INSUFFICIENT"}:
        raise ValueError("Source screening has not naturally reached a terminal state")
    if terminal["completed_candidates"] != config["expected_candidates"] or terminal["contract_sha"] != digest(config):
        raise ValueError("Source full-pool screening binding mismatch")
    if output_root.exists() and any(output_root.iterdir()):
        if (output_root / "terminal.json").exists():
            return json.loads((output_root / "terminal.json").read_text())
        raise ValueError("Gap-reconciliation root must be fresh")
    output_root.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    torch.set_num_threads(int(config.get("threads", 2)))
    source = load_aids_generation_bundle(dataset_dir=config["dataset_dir"], source_csv=config["source_csv"])
    parents = dict(zip(source.parent_ids, source.graphs, strict=True))
    original_binding = json.loads((source_root / "source_input_binding.json").read_text())
    if source.audit() != original_binding:
        raise ValueError("Source changed since the complete screening")
    native = json.loads((Path(config["input_root"]) / "run_manifest.json").read_text())
    native_fingerprint = native["dataset_audit"]["dataset_fingerprint"]
    if source.dataset_fingerprint != native_fingerprint:
        raise ValueError("HPC source dataset differs from the original native generation fingerprint")
    # Prior terminal closes the same immutable full trace and transferred
    # manifest. No second full package/chunk hashing is necessary here.
    predecessor, events = predecessor_index(checked_events(Path(config["input_root"]), verify_hashes=False))
    if events != terminal["trace_events"]:
        raise ValueError("Immutable trace row count differs from previous terminal")
    segments = [json.loads(p.read_text()) for p in sorted((source_root / "segments").glob("segment-*.json"))]
    source_segment_digests = [digest(s) for s in segments]
    gap_indices = {r["candidate_index"] for s in segments for r in s["rows"] if r["state"] == "CACHE_PROVENANCE_GAP"}
    originals = {}
    with (Path(config["input_root"]) / "trace/candidate_action_lineage_index.jsonl").open() as handle:
        for line in handle:
            original = json.loads(line)
            if original["candidate_index"] in gap_indices:
                originals[original["candidate_index"]] = original
    if set(originals) != gap_indices:
        raise ValueError("Gap records are absent from the original complete lineage index")
    cache = {r["canonical_smiles"]: r["rf"] for s in segments for r in s["rows"] if "rf" in r}
    prior_cache_count = len(cache)
    rf = load_oracle_bundle(config["rf_path"])
    if hasattr(rf["model"], "n_jobs"):
        rf["model"].n_jobs = int(config.get("threads", 2))
    repairs = []
    for segment in segments:
        for row in segment["rows"]:
            if row["state"] != "CACHE_PROVENANCE_GAP":
                continue
            before = dict(row)
            original = originals[row["candidate_index"]]
            for key in ("candidate_index", "official_graph_hash", "stable_graph_sha256", "parent_id", "action_count"):
                if original[key] != row[key]:
                    raise ValueError("Gap row binding differs from original lineage index")
            try:
                graph = replay_candidate(original, predecessor, parents)
            except (ValueError, IndexError, KeyError) as exc:
                repairs.append({"candidate_index": row["candidate_index"], "state": "UNRESOLVED", "reason": str(exc)})
                continue
            decoded = decode_representative(graph, dataset="aids", atom_vocabulary=source.atom_vocabulary)
            smiles = decoded["canonical_smiles"]
            mol = Chem.MolFromSmiles(smiles) if decoded["decode_ok"] else None
            row["decode"] = decoded
            if mol is None or len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                row.update(state="CHEM_REJECT", reason=decoded["decode_reason"] or "disconnected_or_dummy")
            else:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles not in cache:
                    cache[smiles] = rf_predict([smiles], rf)[0]
                row.update(canonical_smiles=smiles, rf=cache[smiles], state="RF_TARGET0" if cache[smiles]["prediction"] == 0 else "RF_TARGET_REJECT")
                row.pop("reason", None)
                if row["state"] == "RF_TARGET0":
                    row["graph"] = compact_graph(graph)
            repairs.append({"candidate_index": row["candidate_index"], "old_reason": before["reason"], "state": row["state"], "graph_sha_verified": row["stable_graph_sha256"]})
    counts = Counter(row["state"] for segment in segments for row in segment["rows"])
    for segment in segments:
        atomic_json(output_root / "segments" / f"segment-{segment['start']:09d}.json", segment)
    for name in ("contract.json", "source_input_binding.json", "source_predictions.json"):
        atomic_json(output_root / name, json.loads((source_root / name).read_text()))
    result = dict(terminal, state="POOL_SCREEN_COMPLETE" if not counts["CACHE_PROVENANCE_GAP"] else "EVIDENCE_INSUFFICIENT", counts=dict(counts), unique_chemical_graphs=len(cache), unique_rf_target0=sum(x["prediction"] == 0 for x in cache.values()), corrected_from=str(source_root), correction_reason="NODE_REMOVAL_TARGET_IDS_GLUE_FIX", original_screen_preserved=True, source_segment_digests=source_segment_digests, already_successful_candidates_replayed=0, unique_new_rf_predictions=len(cache) - prior_cache_count, repaired_records=repairs)
    atomic_json(output_root / "terminal.json", result)
    return result
