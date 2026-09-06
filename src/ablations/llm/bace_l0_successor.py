"""BACE L0 evaluation-only adoption of the completed 2560839 train ledger.

No generator, GNN-package replay or train-oracle execution is called here.
The first adoption validates small train artifacts and their committed parents;
all writes are additive under the explicitly authorized HPC project subtree.
"""
from __future__ import annotations

import fcntl
from pathlib import Path
import tarfile
from typing import Any

from src.ablations.llm.at_most_k import POLICY
from src.ablations.llm.corrected_core_gate import _corrective_proof_complete
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, read_json, read_jsonl, sha256_file, stable_sha256,
)

SCHEMA = "bace_l0_at_most_k_train_adoption_v1"
TRAIN_FILES = ("run_manifest.json", "terminal.json", "scored_attempts.jsonl",
               "candidate_pool.jsonl", "candidate_universe.jsonl", "candidate_metrics.json")


def _hash(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("L0_REQUIRES_PHYSICAL_SOURCE_FILE")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size": path.stat().st_size}


def _reopen(identity):
    path = Path(identity["path"])
    if _hash(path) != identity:
        raise ValueError("L0_SOURCE_ARTIFACT_CHANGED")
    return path


def _self_hash(value):
    if value.get("self_sha256") != stable_sha256({k: v for k, v in value.items() if k != "self_sha256"}):
        raise ValueError("L0_SELF_HASH_MISMATCH")


def _train_ledger(source: Path, manifest: dict) -> list[dict]:
    """Validate exact saved science rows, without oracle/OT/LLM replay."""
    from src.ablations.llm.bace_common_downstream import merge_scored_rows
    scored = read_jsonl(source / "scored_attempts.jsonl")
    keys = {(str(r["parent_id"]), int(r["candidate_index"])) for r in scored}
    parents = {p for p, _ in keys}
    if len(scored) != 3088 or len(keys) != 3088 or len(parents) != 386 or keys != {(p, i) for p in parents for i in range(8)}:
        raise ValueError("L0_REQUIRES_COMPLETE_386_BY_8_SCORED_LEDGER")
    if any(r.get("test_loaded") is not False or r.get("calibration_loaded") is not False or r.get("rf_oracle_used") is not False for r in scored):
        raise ValueError("L0_TRAIN_LEDGER_SPLIT_OR_ORACLE_MISMATCH")
    checkpoint_rows = {}
    checkpoints = sorted((source / "parent_checkpoints" / "train").glob("*.json"))
    if len(checkpoints) != 386:
        raise ValueError("L0_TRAIN_PARENT_CHECKPOINT_COUNT_MISMATCH")
    for path in checkpoints:
        state = read_json(path)
        _self_hash(state)
        if state["binding_sha256"] != manifest["binding_sha256"]:
            raise ValueError("L0_TRAIN_CHECKPOINT_BINDING_MISMATCH")
        for row in state["rows"]:
            key = (str(row["parent_id"]), int(row["candidate_index"]))
            if key in checkpoint_rows:
                raise ValueError("L0_DUPLICATE_CHECKPOINT_ATTEMPT")
            checkpoint_rows[key] = row
    if checkpoint_rows != {(str(r["parent_id"]), int(r["candidate_index"])): r for r in scored}:
        raise ValueError("L0_SCORED_LEDGER_CHECKPOINT_MISMATCH")
    merged, universe = merge_scored_rows(scored)
    if merged != read_jsonl(source / "candidate_pool.jsonl") or universe != read_jsonl(source / "candidate_universe.jsonl"):
        raise ValueError("L0_SCORED_CANDIDATE_UNIVERSE_MISMATCH")
    if len(universe) != 15:
        raise ValueError("L0_REAL_SOURCE_UNIVERSE_IS_NOT_15")
    return scored


def prepare(*, source_train_root, corrected_package_receipt, output_root):
    source = Path(source_train_root).resolve(strict=True)
    output = Path(output_root).absolute()
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("L0_OVERLAY_MUST_BE_FRESH_AND_DISJOINT")
    # The completed producer's actual lock is retained read-only during adopt.
    with (source / "writer.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest, terminal = read_json(source / "run_manifest.json"), read_json(source / "terminal.json")
        _self_hash(manifest)
        if (manifest["variant"] != "BRICS_FIXED" or manifest["main_matrix_write"] is not False
                or terminal != {"candidate_padding_used": False, "required_rules": 20,
                    "state": "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_UNIQUE_RULES", "test_loaded": False,
                    "valid_unique_rules": 15}):
            raise ValueError("L0_SOURCE_IS_NOT_THE_PRESERVED_TRAIN_ONLY_FAILURE")
        if any((source / name).exists() for name in ("calibration_pairs.jsonl", "test_pairs.jsonl", "selector_manifest.json", "final_audit.json")):
            raise ValueError("L0_SOURCE_HAS_LATER_EVALUATION_DO_NOT_RECOMPUTE")
        proof = manifest["gnn_independent_core"]
        accepted = read_json(corrected_package_receipt)
        if not _corrective_proof_complete(proof) or not _corrective_proof_complete(accepted):
            raise ValueError("L0_GNN_CORRECTED_ACCEPTANCE_INCOMPLETE")
        for key in ("sha256", "bytes", "corrective_audit_sha256", "independent_science_replay_sha256"):
            if accepted[key] != proof[key]:
                raise ValueError("L0_CORRECTIVE_RECEIPT_MISMATCH")
        scored = _train_ledger(source, manifest)
        sources = {name: _hash(source / name) for name in TRAIN_FILES}
        # This small receipt is already an independent sealed acceptance. No
        # archive scan or extraction is necessary to continue its existing L0.
        receipt = {"schema_version": SCHEMA, "authorization": "USER_PROJECT_OWNER_2026_09_06",
            "selection_policy": POLICY, "source_root": str(source), "source_job_id": "2560839",
            "source_job_preserved": True, "source_files": sources,
            "corrected_package_receipt": _hash(Path(corrected_package_receipt)),
            "train_attempts": len(scored), "parent_count": 386, "vocabulary_size": 472,
            "valid_unique_rules": 15, "generation_repeated": False, "train_oracle_repeated": False,
            "gnn_package_replayed": False, "main_matrix_write": False,
            "allowed_new_science": "FIRST_L0_CALIBRATION_FREEZE_TEST_EVALUATION_ONLY"}
        receipt["self_sha256"] = stable_sha256(receipt)
        output.mkdir(parents=True)
        atomic_json(output / "protocol_overlay.json", receipt)
    return receipt


def load_train_adoption(path):
    overlay = read_json(path)
    _self_hash(overlay)
    if overlay.get("schema_version") != SCHEMA or overlay.get("selection_policy") != POLICY:
        raise ValueError("L0_ADOPTION_PROTOCOL_MISMATCH")
    files = {k: _reopen(v) for k, v in overlay["source_files"].items()}
    manifest = read_json(files["run_manifest.json"])
    _self_hash(manifest)
    proof = read_json(_reopen(overlay["corrected_package_receipt"]))
    if not _corrective_proof_complete(proof) or proof["sha256"] != manifest["gnn_independent_core"]["sha256"]:
        raise ValueError("L0_CORRECTED_PROOF_BINDING_MISMATCH")
    return {"source_manifest": manifest, "scored": read_jsonl(files["scored_attempts.jsonl"]),
            "overlay_identity": _hash(Path(path))}


def run(*, protocol_overlay, portable_input_bundle, gnn_input_bundle, registry_root, output_root,
        resume=False, cpu_threads=8, batch_size=256):
    import os
    from src.ablations.llm.portable_inputs import PortableInputs
    from src.ablations.llm.bace_common_downstream import run_downstream
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise ValueError("L0_SUCCESSOR_IS_CPU_ONLY")
    adoption = load_train_adoption(protocol_overlay)
    portable = PortableInputs(portable_input_bundle)
    receipt = read_json(read_json(protocol_overlay)["corrected_package_receipt"]["path"])
    return run_downstream(task_spec=portable.task_spec_path(),
        candidate_root=portable.root / portable.manifest["brics_root_relative"],
        gnn_input_bundle=gnn_input_bundle, gnn_verified_archive=receipt.get("path", receipt.get("archive")),
        gnn_verified_sha256=receipt["sha256"], registry_root=registry_root, output_root=output_root,
        resume=resume, device="cpu", cpu_threads=cpu_threads, batch_size=batch_size,
        portable_input_bundle=portable.root, train_adoption_overlay=protocol_overlay)


def package(*, science_root, gnn_input_bundle, output_root):
    """Independent small-output audit and compact package; no model inference."""
    from src.ablations.llm.at_most_k import explanation_metrics, select_calibration
    from src.ablations.gnn import cpu_evaluation as evaluation
    science, output = Path(science_root).resolve(strict=True), Path(output_root).absolute()
    if output.exists() or science in output.parents or output in science.parents:
        raise ValueError("L0_PACKAGE_MUST_BE_FRESH_AND_DISJOINT")
    with (science / "writer.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audit = read_json(science / "final_audit.json")
        if audit.get("state") != "PASS" or audit.get("main_matrix_write") is not False:
            raise ValueError("L0_FINAL_SCIENTIFIC_AUDIT_NOT_PASS")
        for rel, digest in audit["files"].items():
            path = science / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or sha256_file(path) != digest:
                raise ValueError("L0_FINAL_OUTPUT_CHANGED")
        freeze = read_json(science / "selector_manifest.json")
        _self_hash(freeze)
        if freeze.get("selection_policy") != POLICY or freeze["K_EFFECTIVE"] != 15 or freeze["test_loaded"] is not False:
            raise ValueError("L0_AT_MOST_K_FREEZE_MISMATCH")
        selected = read_json(science / "selected_rules.json")["rules"]
        if [r["candidate_id"] for r in selected] != freeze["ordered_rule_ids"] or len(set(freeze["ordered_rule_ids"])) != 15:
            raise ValueError("L0_SELECTED_RULES_PADDED_OR_CHANGED")
        metrics = read_json(science / "heldout_test_metrics.json")
        if metrics["K_EFFECTIVE"] != 15 or len(metrics["prefix_rows"]) != 20:
            raise ValueError("L0_METRICS_K_CONTRACT_MISMATCH")
        for row in metrics["prefix_rows"][15:]:
            if row["ccrcov_theta_star"] != metrics["prefix_rows"][14]["ccrcov_theta_star"] or row["effective_k"] != 15:
                raise ValueError("L0_METRICS_FALSE_PADDED_PREFIX")
        # Replay only the small saved matrices/selector, never model or OT.
        bundle = Path(gnn_input_bundle).resolve(strict=True)
        manifest = read_json(bundle / "bundle_manifest.json")
        run_manifest = read_json(science / "run_manifest.json")
        if sha256_file(bundle / "bundle_manifest.json") != run_manifest["bundle_sha256"]:
            raise ValueError("L0_PACKAGE_FROZEN_BUNDLE_MISMATCH")
        selector = evaluation.frozen_selector(bundle, manifest)
        universe = read_jsonl(science / "candidate_universe.jsonl")
        calibration = evaluation.matrix_from_pairs(
            read_json(science / "calibration_cohort_manifest.json")["parent_ids"], universe,
            read_jsonl(science / "calibration_pairs.jsonl"), root=science, split="calibration")
        order, _ = select_calibration(calibration, selector)
        if [universe[i]["candidate_id"] for i in order] != freeze["ordered_rule_ids"]:
            raise ValueError("L0_PACKAGE_CALIBRATION_SELECTOR_REPLAY_MISMATCH")
        test = evaluation.matrix_from_pairs(read_json(science / "test_cohort_manifest.json")["parent_ids"],
            selected, read_jsonl(science / "test_pairs.jsonl"), root=science, split="test")
        if explanation_metrics(test, range(len(selected)), selector["thresholds"]) != metrics:
            raise ValueError("L0_PACKAGE_NUMERIC_METRICS_REPLAY_MISMATCH")
        output.mkdir(parents=True)
        inventory = {rel: _hash(science / rel) for rel in sorted(audit["files"])}
        inventory["final_audit.json"] = _hash(science / "final_audit.json")
        independent = {"state": "PASS", "schema_version": "bace_l0_at_most_k_package_v1",
            "selection_policy": POLICY, "K_EFFECTIVE": 15, "source_root": str(science),
            "files": inventory, "model_inference_repeated": False, "ot_recomputed": False,
            "main_matrix_write": False, "source_failure_preserved": True}
        atomic_json(output / "independent_package_audit.json", independent)
        archive = output / "bace_l0_at_most_k.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for rel in inventory:
                handle.add(science / rel, arcname="result/" + rel, recursive=False)
            handle.add(output / "independent_package_audit.json", arcname="independent_package_audit.json")
        receipt = {"state": "PASS", "path": str(archive), "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size, "K_EFFECTIVE": 15, "main_matrix_write": False,
            "independent_audit_sha256": sha256_file(output / "independent_package_audit.json")}
        atomic_json(output / "result_package.json", receipt)
        return receipt
