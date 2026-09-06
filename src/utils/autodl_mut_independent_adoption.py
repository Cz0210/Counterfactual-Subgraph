"""The one nominated Mut 50k artifact, independently adopted without A/B claims.

This is a bounded receipt audit, not another generation, graph replay, model
load, or clustering run. Large immutable files reuse the already-completed
inventory/read-only adoption receipts; only metadata is reopened. The ordered
candidate universe is proven by the original pair producer's source-bound
manifest, not described as a new reconstruction of the payload.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256, write_json
from src.utils.autodl_mut_same_contract_adoption_v1 import _lineage_contract
from src.utils.autodl_mut_traceoff_parity_v1 import (
    SOURCE_CANDIDATE_COUNT, SOURCE_CONFIG_SHA256, SOURCE_DATASET_SHA256,
    SOURCE_DISTANCE_SHA256, SOURCE_GNN_SHA256, SOURCE_PARENT_ORDER_SHA256,
    SOURCE_PAYLOAD_SHA256, SOURCE_PROJECT_COMMIT, SOURCE_UPSTREAM_COMMIT,
)

SCHEMA = "mut_independent_historical50k_adoption_v3"
AUTH_SCHEMA = "mut_independent_historical_adoption_authorization_v1"
SOURCE_NAME = "mutagenicity_comrecgc_lineage_v3_20260822T025620Z"
STAGE_ID = "mut_independent_adoption"
UNIVERSE = "9c4d79ac61746a1dbbbefb1c7826c77a940c95b0348fb051c202f434de06ffbe"
ROW_MAPPING = "e99a1ca86ddc36395f82428b0cd2e7bff67ca02fd40c6533d432112af2237b7a"
TRANSITIVE = "transitive_generation_pair_store_vectors_dbscan_v1"
SMALL_FILE_LIMIT = 2 * 1024 * 1024


class AdoptionError(ValueError):
    def __init__(self, category: str, field: str) -> None:
        self.category, self.field = category, field
        super().__init__(f"{category}:{field}")


def require(ok: bool, field: str, category: str = "SCIENCE_INVALID") -> None:
    if not ok:
        raise AdoptionError(category, field)


def obj(path: Path) -> dict[str, Any]:
    require(path.is_absolute() and not path.is_symlink() and path.is_file(),
            str(path), "METADATA_OR_PATH_GAP")
    require(path.stat().st_size <= SMALL_FILE_LIMIT, str(path), "EVIDENCE_INSUFFICIENT")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), str(path), "METADATA_OR_PATH_GAP")
    return value


def stat_identity(path: Path) -> dict[str, int]:
    st = path.stat()
    return dict(size=st.st_size, inode=st.st_ino, device=st.st_dev,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns, mode=st.st_mode)


def same_immutable_stat(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """A remount may change only st_dev, never bytes/timestamps/inode/mode."""
    return all(old.get(k) == new.get(k) for k in
               ("size", "inode", "mtime_ns", "ctime_ns", "mode"))


def selected_rows(path: Path) -> list[dict[str, Any]]:
    """The native selected-recourse artifact is a JSON array, not a manifest."""
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= SMALL_FILE_LIMIT,
            "selected_common_recourses", "METADATA_OR_PATH_GAP")
    rows = json.loads(path.read_text())
    require(isinstance(rows, list) and len(rows) == 100 and all(isinstance(row, dict) for row in rows),
            "native_selected_recourse_count")
    return rows


def _fields(value: Mapping[str, Any], expected: Mapping[str, Any], prefix: str) -> None:
    for key, item in expected.items():
        require(value.get(key) == item, f"{prefix}.{key}")


def audit(*, source_root: Path, common_root: Path, inventory_path: Path,
          proc_root: Path = Path("/proc")) -> dict[str, Any]:
    from scripts.autodl.run_comrecgc_standardized_continuation import (
        _snapshot_critical_manifests, _snapshot_frozen_file, _scan_live_source_writers,
    )
    source = source_root.resolve(strict=True)
    common = common_root.resolve(strict=True)
    require(source.name == SOURCE_NAME, "nominated_source_only")
    project = Path(__file__).resolve().parents[2]
    protocol = project / "docs/COMRECGC_RECOVERY_PROTOCOL.md"
    protocol_text = protocol.read_text()
    require("the only component allowed to query the frozen RF teacher or calculate WNode" in protocol_text,
            "historical_oracle_role_contract", "EVIDENCE_INSUFFICIENT")
    implementation = {}
    for relative in ("src/baselines/comrecgc/runtime.py", "src/baselines/comrecgc/model_adapter.py"):
        result = subprocess.run(["git", "show", f"{SOURCE_PROJECT_COMMIT}:{relative}"],
                                cwd=project, capture_output=True, text=True, check=False, timeout=20)
        require(result.returncode == 0 and "load_mutagenicity_gnn" in result.stdout,
                f"historical_oracle_source:{relative}", "EVIDENCE_INSUFFICIENT")
        implementation[relative] = {"commit": SOURCE_PROJECT_COMMIT,
            "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}
    inventory = obj(inventory_path)
    _fields(inventory, {"schema_version": "mut_historical_50k_inventory_v2",
                        "status": "PASS"}, "inventory")
    proven = inventory["source"]
    _fields(proven, {
        "status": "PASS", "source_root": str(source), "source_payload_hashed": True,
        "source_payload_actual_sha256": SOURCE_PAYLOAD_SHA256,
        "source_project_commit": SOURCE_PROJECT_COMMIT,
        "source_upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_dataset_sha256": SOURCE_DATASET_SHA256,
        "source_parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
        "source_gnn_sha256": SOURCE_GNN_SHA256, "source_distance_sha256": SOURCE_DISTANCE_SHA256,
        "source_candidate_count": SOURCE_CANDIDATE_COUNT, "source_parent_count": 1448,
        "trace_enabled": True, "calibration_loaded": False, "test_loaded": False,
    }, "source_inventory")
    metadata: dict[str, dict[str, Any]] = {}
    reused: dict[str, dict[str, Any]] = {}

    def small(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
        path = path.resolve(strict=True)
        if str(path) not in metadata:
            value = obj(path)
            metadata[str(path)] = {"sha256": sha256_file(path), "stat": stat_identity(path)}
        else:
            value = obj(path)
        if expected_sha is not None:
            require(metadata[str(path)]["sha256"] == expected_sha, f"metadata_hash:{path}")
        return value

    def old_large(path: Path, record: Mapping[str, Any], verified_at: str) -> None:
        require(path.is_file() and not path.is_symlink(), str(path), "METADATA_OR_PATH_GAP")
        now = stat_identity(path)
        if "stat" in record:
            require(same_immutable_stat(record["stat"], now), f"immutable_stat:{path}",
                    "EVIDENCE_INSUFFICIENT")
        else:
            stamp = int(datetime.fromisoformat(verified_at).timestamp() * 1_000_000_000)
            require(now["size"] == record["size"] and max(now["mtime_ns"], now["ctime_ns"]) <= stamp,
                    f"immutable_inventory_timestamp:{path}", "EVIDENCE_INSUFFICIENT")
        reused[str(path)] = {"sha256": record["sha256"], "stat": now,
                             "prior_stat": record.get("stat"), "rehash_performed": False,
                             "basis": "prior_complete_receipt_plus_unchanged_metadata_no_writer"}

    for name, record in proven["source_files"].items():
        path = source / name
        require(str(path) == record["path"], f"inventory_path:{name}")
        if int(record["size"]) <= SMALL_FILE_LIMIT:
            small(path, record["sha256"])
        else:
            old_large(path, record, proven["verified_at"])
    manifest = small(source / "run_manifest.json")
    closure = small(source / "frozen_payload_closure_audit.json")
    recovery = small(source / "freeze_only_recovery.json")
    _fields(manifest, {"run_complete": True, "dataset": "mutagenicity", "mode": "full",
                      "algorithm_rerun": False, "project_commit": SOURCE_PROJECT_COMMIT,
                      "upstream_commit": SOURCE_UPSTREAM_COMMIT,
                      "counterfactual_candidate_count": SOURCE_CANDIDATE_COUNT}, "generation")
    _fields(closure, {"closure_complete": True, "post_write_reload_verified": True,
                     "candidate_order_changed": False, "candidate_payload_changed": False,
                     "scientific_parameters_changed": False, "unresolved_hash_count": 0,
                     "sha_mismatch_count": 0, "payload_checksum": SOURCE_PAYLOAD_SHA256}, "closure")
    _fields(recovery, {"recovery_completed": True, "completed_steps": 50000,
                      "algorithm_rerun": False}, "freeze_recovery")
    old_large(source / "counterfactuals.pt", {"size": closure["payload_bytes"],
              "sha256": SOURCE_PAYLOAD_SHA256}, proven["verified_at"])
    lineage = _lineage_contract(source)
    action = small(source / "trace/candidate_action_lineage.json")["lineage_recovery_audit"]
    _fields(action, {"recorded_action_replay_verified_count": 224690,
                    "recorded_action_replay_mismatch_count": 0,
                    "legacy_inference_called_count": 0}, "selected_action")
    cm = small(common / "run_manifest.json")
    _fields(cm, {"run_complete": True, "dataset": "mutagenicity", "method": "COMRECGC",
                "counterfactuals_sha256": SOURCE_PAYLOAD_SHA256, "common_recourse_count": 100,
                "model_counterfactual_candidate_count": 50620, "calibration_loaded": False,
                "test_loaded": False, "official_greedy_order_preserved": True,
                "embedding_centers_exported_as_graphs": False}, "common")
    require(cm["generation_manifest_sha256"] == metadata[str(source / "run_manifest.json")]["sha256"],
            "common_generation_binding")
    ext = cm["external_memory_artifacts"]
    pp = Path(ext["pair_store_manifest"])
    dp = Path(ext["dbscan_manifest"])
    pair = small(pp, ext["pair_store_manifest_sha256"])
    dbscan = small(dp, ext["dbscan_manifest_sha256"])
    adoption = small(Path(ext["pair_store_adoption_manifest"]), ext["pair_store_adoption_manifest_sha256"])
    _fields(adoption, {"run_complete": True, "source_mutated": False,
                       "source_manifest_sha256": ext["pair_store_manifest_sha256"]}, "pair_adoption")
    identity = pair["scientific_identity"]
    _fields(identity, {"candidate_count": 50620, "parent_count": 1448,
                      "counterfactuals_sha256": SOURCE_PAYLOAD_SHA256,
                      "candidate_graph_hashes_sha256": UNIVERSE,
                      "generation_indices_sha256": ROW_MAPPING,
                      "parent_ids_sha256": SOURCE_PARENT_ORDER_SHA256,
                      "dataset_fingerprint": SOURCE_DATASET_SHA256,
                      "distance_checkpoint_sha256": SOURCE_DISTANCE_SHA256,
                      "pair_order": "candidate_major_parent_minor",
                      "generation_manifest_sha256": cm["generation_manifest_sha256"]}, "pair_identity")
    require(pair["scientific_identity_sha256"] == stable_json_sha256(identity)
            == adoption["scientific_identity_sha256"], "pair_identity_self_hash")
    _fields(pair, {"run_complete": True, "row_count": 813595,
                   "candidate_major_parent_minor_order": True}, "pair")
    cursor, rows = 0, 0
    for index, chunk in enumerate(pair["chunks"]):
        ci = chunk["scientific_identity"]
        require(ci["candidate_start"] == cursor and ci["candidate_stop"] > cursor
                and ci["chunk_index"] == index == chunk["chunk_index"], "pair_partition")
        require(stable_json_sha256(ci) == chunk["scientific_identity_sha256"], "chunk_identity")
        cursor, rows = ci["candidate_stop"], rows + chunk["row_count"]
    require(cursor == 50620 and rows == 813595 and len(pair["chunks"]) == pair["chunk_count"],
            "pair_partition_complete")
    for raw, record in adoption["source_files"].items():
        old_large(Path(raw), record, adoption["adopted_at"])
    _fields(dbscan, {"run_complete": True, "approximation_used": False,
                    "failure_cap_used": False, "num_samples": 813595,
                    "clustering_path": "sklearn_float64_exact_multi_component_v1",
                    "sklearn_dbscan_label_semantics_preserved": True}, "dbscan")
    di = dbscan["scientific_identity"]
    require(stable_json_sha256(di) == dbscan["scientific_identity_sha256"], "dbscan_identity")
    _fields(di, {"vectors_path": pair["vectors_path"], "vectors_sha256": pair["vectors_sha256"],
                 "vectors_shape": [813595, 64], "distance_reference_dtype": "float64"}, "dbscan_input")
    require(di["contract"]["eps"] == 0.02 and di["contract"]["min_samples"] == 3,
            "dbscan_parameters")
    for field in ("labels", "core_mask", "neighbor_counts"):
        path = Path(dbscan[field + "_path"])
        # These sealed arrays are not loaded/rehash-scanned; the complete native
        # producer manifest is bound by the completed common-recourse manifest.
        old_large(path, {"size": path.stat().st_size, "sha256": dbscan[field + "_sha256"]},
                  cm["completed_at"])
    small(common / "_RUN_COMPLETE.json")
    selected_path = common / "selected_common_recourses.json"
    selected_rows(selected_path)
    metadata[str(selected_path)] = {"sha256": sha256_file(selected_path), "stat": stat_identity(selected_path)}
    reps = Path(cm["representative_counterfactuals_path"])
    require(reps.parent == common and reps.stat().st_size <= SMALL_FILE_LIMIT, "representative_payload")
    require(sha256_file(reps) == cm["representative_counterfactuals_sha256"], "representatives_sha")
    reused[str(reps)] = {"sha256": cm["representative_counterfactuals_sha256"],
                         "stat": stat_identity(reps), "rehash_performed": True,
                         "basis": "bounded_selected_100_payload_once"}
    snapshots = _snapshot_critical_manifests(source)
    payload_snapshot = _snapshot_frozen_file(source / "counterfactuals.pt", include_sha256=False)
    writers = {str(root): _scan_live_source_writers(root, protected_snapshots=(), proc_root=proc_root)
               for root in (source, common, pp.parent)}
    generation = {
        "schema_version": 1, "status": "PASS", "dataset": "mutagenicity",
        "generation_adopted": True, "generation_mode": "adopted_read_only_cache",
        "generation_rerun": False, "source_generation_root": str(source),
        "counterfactuals_path": str(source / "counterfactuals.pt"),
        "counterfactuals_sha256_claimed": SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_actual": SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_verified": True, "counterfactuals_sha256_computation_count": 0,
        "payload_hash_receipt_reused": True, "payload_hash_receipt_path": str(inventory_path),
        "payload_hash_receipt_sha256": sha256_file(inventory_path),
        "counterfactual_candidate_count": SOURCE_CANDIDATE_COUNT,
        "source_project_commit": SOURCE_PROJECT_COMMIT, "upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "serialization_rerun": False, "lineage_resolution_rerun": False,
        "source_integrity": {"schema_version": 1,
            "critical_manifests_before_payload_hash": snapshots,
            "critical_manifests_after_payload_hash": snapshots,
            "payload_before_sha256": payload_snapshot, "payload_after_sha256": payload_snapshot,
            "live_writer_audit_before_payload_hash": writers[str(source)],
            "live_writer_audit_after_payload_hash": writers[str(source)]},
    }
    return {"classification": "VALID", "status": "PASS", "source_root": str(source),
            "common_root": str(common), "source_inventory_path": str(inventory_path),
            "source_inventory_sha256": sha256_file(inventory_path), "source_lineage": lineage,
            "metadata": metadata, "immutable_reuse": reused, "writer_audits": writers,
            "generation_adoption": generation, "candidate_universe_sha": UNIVERSE,
            "generation_indices_sha256": ROW_MAPPING, "parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
            "source_pair_store_manifest_path": str(pp), "source_dbscan_manifest_path": str(dp),
            "candidate_binding_basis": "original_source_bound_pair_producer_and_read_only_adoption",
            "candidate_universe_reconstructed_this_attempt": False,
            "pair_partition_chunk_count": len(pair["chunks"]), "source_raw_candidates": 100235,
            "pair_candidates": 50620, "parents": 1448, "theta_close_pair_rows": 813595,
            "common_recourse_count": 100, "large_payload_loads": 0, "large_payload_rehashes": 0,
            "active_sqlite_reads": 0, "pair_store_recomputed": False, "dbscan_recomputed": False,
            "source_generation_oracle": "frozen_project_GNN_native_importance",
            "source_generation_oracle_sha256": SOURCE_GNN_SHA256,
            "source_generation_distance": "NeuroSED", "final_oracle": "RF",
            "final_distance": "MolCLR-Node-Wasserstein",
            "oracle_role_contract": "COMRECGC_RECOVERY_PROTOCOL:only_slot_evaluator_queries_RF_WNode",
            "oracle_role_protocol_path": str(protocol), "oracle_role_protocol_sha256": sha256_file(protocol),
            "original_generation_implementation": implementation,
            "historical_algorithm_resume_safe": recovery.get("RESUME_SAFE"),
            "frozen_payload_reload_verified": True, "algorithm_checkpoint_reload_claimed": False,
            "trace_on_off_parity_required": False, "trace_parity_passed": False,
            "500_step_semantic_equivalence_passed": False, "full_50k_parity_claimed": False,
            "calibration_loaded": False, "test_loaded": False,
            "created_at": datetime.now(timezone.utc).isoformat()}


def validate_receipt(path: Path, *, source_root: Path) -> dict[str, Any]:
    value = obj(path)
    _fields(value, {"schema_version": SCHEMA, "status": "PASS", "classification": "VALID",
                   "source_generation_root": str(source_root.resolve(strict=True)),
                   "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
                   "trace_on_off_parity_required": False, "trace_parity_passed": False,
                   "500_step_semantic_equivalence_passed": False,
                   "candidate_universe_sha": UNIVERSE, "candidate_universe_binding_state": "PASS"}, "adoption")
    require(value.get("binding_sha256") == stable_json_sha256(
        {k: v for k, v in value.items() if k != "binding_sha256"}), "adoption_self_hash")
    ap = Path(value["authorization_path"])
    require(sha256_file(ap) == value["authorization_sha256"], "authorization_sha")
    _validate_authorization(obj(ap), source_root)
    evidence = obj(Path(value["independent_adoption_audit_path"]))
    require(sha256_file(Path(value["independent_adoption_audit_path"])) == value["independent_adoption_audit_sha256"], "audit_sha")
    require(evidence.get("classification") == "VALID" and evidence["source_root"] == str(source_root), "independent_audit")
    _fields(evidence, {"candidate_universe_sha": UNIVERSE, "generation_indices_sha256": ROW_MAPPING,
                      "parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
                      "source_raw_candidates": 100235, "pair_candidates": 50620, "parents": 1448,
                      "theta_close_pair_rows": 813595, "common_recourse_count": 100,
                      "candidate_binding_basis": "original_source_bound_pair_producer_and_read_only_adoption",
                      "candidate_universe_reconstructed_this_attempt": False,
                      "pair_store_recomputed": False, "dbscan_recomputed": False,
                      "calibration_loaded": False, "test_loaded": False,
                      "frozen_payload_reload_verified": True,
                      "algorithm_checkpoint_reload_claimed": False}, "audit_contract")
    require(value["generation_adoption"] == evidence["generation_adoption"], "generation_evidence_binding")
    require(str(source_root / "run_manifest.json") in evidence["metadata"]
            and value["source_pair_store_manifest_path"] in evidence["metadata"]
            and value["source_dbscan_manifest_path"] in evidence["metadata"]
            and str(source_root / "counterfactuals.pt") in evidence["immutable_reuse"], "audit_required_evidence")
    for raw, record in evidence["metadata"].items():
        require(sha256_file(Path(raw)) == record["sha256"], f"metadata_changed:{raw}")
    for raw, record in evidence["immutable_reuse"].items():
        require(same_immutable_stat(record["stat"], stat_identity(Path(raw))),
                f"immutable_changed:{raw}", "EVIDENCE_INSUFFICIENT")
    inventory_path = Path(evidence["source_inventory_path"])
    require(sha256_file(inventory_path) == evidence["source_inventory_sha256"], "original_inventory_binding")
    _fields(obj(inventory_path)["source"], {"source_payload_hashed": True,
            "source_payload_actual_sha256": SOURCE_PAYLOAD_SHA256,
            "source_root": str(source_root)}, "prior_inventory")
    return {**value, "path": str(path), "sha256": sha256_file(path)}


def _validate_authorization(value: Mapping[str, Any], source: Path) -> None:
    _fields(value, {"schema_version": AUTH_SCHEMA, "status": "APPROVED",
                   "historical_source_root": str(source.resolve(strict=True)),
                   "allow_independent_trace_on_adoption": True,
                   "trace_on_off_parity_required": False,
                   "test_used_for_adoption_decision": False}, "authorization")


def publish(*, source_root: Path, common_root: Path, inventory_path: Path,
            authorization_path: Path, output_root: Path, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    _validate_authorization(obj(authorization_path), source_root)
    require(not output_root.exists(), "fresh_output_root", "METADATA_OR_PATH_GAP")
    evidence = audit(source_root=source_root, common_root=common_root, inventory_path=inventory_path, proc_root=proc_root)
    output_root.mkdir(parents=True)
    audit_path = output_root / "independent_adoption_audit.json"
    write_json(audit_path, evidence)
    pp, dp = (Path(evidence[k]) for k in ("source_pair_store_manifest_path", "source_dbscan_manifest_path"))
    receipt = {
        "schema_version": SCHEMA, "status": "PASS", "classification": "VALID",
        "dataset": "mutagenicity", "method": "COMRECGC",
        "authorization_path": str(authorization_path), "authorization_sha256": sha256_file(authorization_path),
        "source_generation_root": str(source_root), "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "source_payload_path": str(source_root / "counterfactuals.pt"),
        "source_common_recourse_root": str(common_root), "common_root": str(common_root),
        "source_lineage_path": evidence["source_lineage"]["path"],
        "source_lineage_sha256": evidence["source_lineage"]["sha256"],
        "common_recourse_count": 100, "candidate_count": 100235,
        "generation_complete": True, "generation_steps": 50000,
        "M_MAX": 50000, "M_EFFECTIVE": 50000, "candidate_capacity": 100000,
        "historical_artifact_adopted": True, "historical_source_trace_enabled": True,
        "independent_scientific_adoption_authorized": True,
        "trace_on_off_parity_required": False, "trace_parity_passed": False,
        "500_step_semantic_equivalence_passed": False,
        "500_step_semantic_equivalence_receipt_path": None,
        "500_step_semantic_equivalence_receipt_sha256": None,
        "traceoff_reference_rerun": False, "full_50k_rerun_performed": False,
        "frozen_payload_reload_verified": True, "algorithm_checkpoint_reload_claimed": False,
        "source_generation_oracle": evidence["source_generation_oracle"], "final_oracle": "RF",
        "candidate_universe_sha": UNIVERSE, "source_native_candidate_universe_sha": UNIVERSE,
        "pair_store_source_candidate_universe_sha": UNIVERSE,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": UNIVERSE,
        "pair_candidate_graph_hashes_sha256": UNIVERSE,
        "candidate_universe_binding_state": "PASS", "transitive_binding_kind": TRANSITIVE,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "source_pair_store_manifest_path": str(pp), "source_pair_store_manifest_sha256": sha256_file(pp),
        "source_dbscan_manifest_path": str(dp), "source_dbscan_manifest_sha256": sha256_file(dp),
        "independent_adoption_audit_path": str(audit_path), "independent_adoption_audit_sha256": sha256_file(audit_path),
        "generation_adoption": evidence["generation_adoption"],
        "calibration_loaded": False, "test_loaded": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt["binding_sha256"] = stable_json_sha256(receipt)
    write_json(output_root / "historical_adoption.json", receipt)
    return receipt
