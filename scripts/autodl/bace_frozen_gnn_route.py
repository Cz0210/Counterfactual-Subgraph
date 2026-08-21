#!/usr/bin/env python3
"""Execute the honest B6 preflight and fail-closed B7--B14 route probes.

The historical BACE PPO, candidate, WNode, selector, and final-evaluation
entrypoints are bound to a Morgan-RF teacher.  This driver never calls those
scientific kernels.  B6 instead exercises the complete calibrated GNN scoring
boundary on real connected deletion records from B5 and then records the stage
as BLOCKED because no PPO update occurred.  Later stages publish structured
blockers until their dataset-specific GNN integrations exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from src.chem.hard_deletion import enumerate_connected_hard_deletions
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeaturizer,
)
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.oracles.gnn_oracle import sha256_file, verify_checkpoint_bundle
from src.oracles.oracle_factory import build_oracle
from src.utils.autodl_runtime import atomic_write_json, fsync_directory, utc_now


SCIENTIFIC_BLOCKED_EXIT_CODE = 78
B6_STAGE = "B6_PPO_SMOKE"
BLOCKED_STAGES = (
    "B7_PPO_FULL",
    "B8_POOL_BASE",
    "B9_POOL_HIGHTEMP",
    "B10_POOL_MERGED",
    "B11_CROSS_PARENT_VERIFIED",
    "B12_SELECTOR",
    "B13_FINAL_EVAL",
    "B14_FROZEN",
)


STAGE_BLOCKERS: dict[str, dict[str, Any]] = {
    "B7_PPO_FULL": {
        "code": "BLOCKED_MISSING_GNN_PPO_INTEGRATION",
        "missing_interface": (
            "A stable PPO reward adapter that accepts one loaded GNNOracle/BaseOracle, "
            "batches parent/residual inference, and logs multiclass-safe CF semantics."
        ),
        "legacy_entrypoints": [
            "scripts/train_ppo_stable.py",
            "scripts/train_mutagenicity_ppo_stable.py",
            "src/rewards/counterfactual_oracle.py",
            "src/rewards/teacher_semantic.py",
        ],
        "legacy_reason": (
            "The stable loop constructs CounterfactualTeacherScorer and "
            "TeacherSemanticScorer around a legacy teacher bundle; the BACE GNN "
            "oracle cannot be injected without changing reward semantics."
        ),
        "required_next_outputs": [
            "ppo_manifest.json",
            "oracle_provenance.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        ],
    },
    "B8_POOL_BASE": {
        "code": "BLOCKED_MISSING_GNN_CANDIDATE_SCORER_INTEGRATION",
        "missing_interface": (
            "Candidate generation that accepts the frozen BACE PPO adapter and "
            "scores generated fragments with the same loaded calibrated GNN."
        ),
        "legacy_entrypoints": [
            "scripts/generate_full_candidate_pool.py",
            "src/eval/full_candidate_pool.py",
        ],
        "legacy_reason": (
            "The existing pool generator requires teacher_path and constructs "
            "TeacherSemanticScorer, CounterfactualTeacherScorer, and ChemRLRewarder "
            "from the historical teacher bundle."
        ),
        "required_next_outputs": [
            "candidate_pool.jsonl",
            "pool_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B9_POOL_HIGHTEMP": {
        "code": "BLOCKED_MISSING_GNN_HIGHTEMP_POOL_INTEGRATION",
        "missing_interface": (
            "The GNN-clean candidate scorer from B8, parameterized with the frozen "
            "temperature=0.7, top_p=0.9, num_return_sequences=4 contract."
        ),
        "legacy_entrypoints": [
            "scripts/generate_full_candidate_pool.py",
            "src/eval/full_candidate_pool.py",
        ],
        "legacy_reason": (
            "Changing sampling parameters does not remove the legacy teacher scorer "
            "or establish GNN reward provenance."
        ),
        "required_next_outputs": [
            "candidate_pool.jsonl",
            "pool_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B10_POOL_MERGED": {
        "code": "BLOCKED_MISSING_GNN_POOL_MERGE_INTEGRATION",
        "missing_interface": (
            "A provenance gate proving that two passing GNN-scored B8/B9 pools "
            "share the same PPO and classifier identities before invoking the "
            "oracle-neutral deterministic merge kernel."
        ),
        "legacy_entrypoints": [
            "scripts/merge_candidate_pools.py",
            "src/eval/candidate_pool_merge.py",
        ],
        "legacy_reason": (
            "The deterministic merge kernel is oracle-neutral, but it cannot sanitize "
            "or promote RF-contaminated or provenance-unknown pool inputs."
        ),
        "required_next_outputs": [
            "candidate_pool.jsonl",
            "merge_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B11_CROSS_PARENT_VERIFIED": {
        "code": "BLOCKED_MISSING_GNN_CROSS_PARENT_VERIFICATION_INTEGRATION",
        "missing_interface": (
            "Calibration candidate x parent x match-instance hard-deletion verification "
            "using batched GNN inference plus the independent MolCLR WNode encoder."
        ),
        "legacy_entrypoints": [
            "src/eval/wnode_action_matrix.py",
            "src/eval/mutagenicity_wnode_matrix.py",
        ],
        "legacy_reason": (
            "The BACE action-matrix path still calls predict_with_teacher and records a "
            "teacher identity; its saved matrices are RF_CONTAMINATED."
        ),
        "required_next_outputs": [
            "pair_details.csv",
            "action_matrix.npz",
            "matrix_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B12_SELECTOR": {
        "code": "BLOCKED_MISSING_GNN_SELECTOR_MATRIX_CONTRACT",
        "missing_interface": (
            "A GNN-verification matrix adapter for the existing calibration-only "
            "prefix-aware selector, with no legacy teacher identity."
        ),
        "legacy_entrypoints": ["src/eval/wnode_prefix_selector.py"],
        "legacy_reason": (
            "The selector math is reusable only after its input schema is rebound to "
            "the new GNN matrix; current manifests preserve historical teacher identity."
        ),
        "required_next_outputs": [
            "selected_top20.json",
            "frozen_selection_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B13_FINAL_EVAL": {
        "code": "BLOCKED_MISSING_GNN_HELDOUT_EVALUATION_INTEGRATION",
        "missing_interface": (
            "A one-shot held-out test evaluator that consumes the frozen B12 ordering, "
            "uses the B4 GNN and MolCLR separately, and never refits thresholds."
        ),
        "legacy_entrypoints": [
            "scripts/evaluate_bace_method.py",
            "scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py",
        ],
        "legacy_reason": (
            "The historical BACE evaluator loads the Morgan-RF teacher; reusing its test "
            "artifacts would violate both oracle provenance and held-out semantics."
        ),
        "required_next_outputs": [
            "final_metrics.json",
            "pair_details.csv",
            "test_evaluation_manifest.json",
            "oracle_provenance.json",
        ],
    },
    "B14_FROZEN": {
        "code": "BLOCKED_MISSING_GNN_FINAL_FREEZE_INTEGRATION",
        "missing_interface": (
            "A final provenance/freeze gate bound to a passing held-out B13 result, "
            "selected calibration rules, classifier identity, and MolCLR identity."
        ),
        "legacy_entrypoints": ["scripts/audit_bace_paper_artifacts.py"],
        "legacy_reason": (
            "Historical final artifacts and their audit are bound to the old BACE route; "
            "no GNN-clean B13 artifact exists to freeze."
        ),
        "required_next_outputs": [
            "FINAL_PASS.json",
            "freeze_manifest.json",
            "oracle_provenance.json",
        ],
    },
}


PREDECESSOR_OUTPUT_CONTRACT: dict[str, tuple[str, ...]] = {
    "B7_PPO_FULL": ("ppo_smoke_manifest.json", "oracle_provenance.json"),
    "B8_POOL_BASE": ("ppo_manifest.json", "oracle_provenance.json"),
    "B9_POOL_HIGHTEMP": ("candidate_pool.jsonl", "pool_manifest.json"),
    "B10_POOL_MERGED": ("candidate_pool.jsonl", "pool_manifest.json"),
    "B11_CROSS_PARENT_VERIFIED": ("candidate_pool.jsonl", "merge_manifest.json"),
    "B12_SELECTOR": ("matrix_manifest.json", "oracle_provenance.json"),
    "B13_FINAL_EVAL": ("selected_top20.json", "frozen_selection_manifest.json"),
    "B14_FROZEN": ("final_metrics.json", "test_evaluation_manifest.json"),
}

LEGACY_ROUTE_AUDIT: tuple[dict[str, Any], ...] = (
    {
        "component": "ChemLLM base proposer",
        "entrypoints": ["pretrained_models/ChemLLM-7B-Chat"],
        "classification": "ORACLE_NEUTRAL",
        "reuse_policy": "proposal_initialization_only_with_new_gnn_scoring",
    },
    {
        "component": "historical BACE PPO/checkpoints",
        "entrypoints": ["outputs/hpc/bace"],
        "classification": "RF_CONTAMINATED",
        "reuse_policy": "diagnosis_only",
    },
    {
        "component": "stable PPO reward",
        "entrypoints": [
            "scripts/train_ppo_stable.py",
            "src/rewards/counterfactual_oracle.py",
            "src/rewards/teacher_semantic.py",
        ],
        "classification": "RF_CONTAMINATED",
        "reuse_policy": "requires_new_base_oracle_injection",
    },
    {
        "component": "candidate generation and scoring",
        "entrypoints": [
            "scripts/generate_full_candidate_pool.py",
            "src/eval/full_candidate_pool.py",
        ],
        "classification": "RF_CONTAMINATED",
        "reuse_policy": "requires_new_gnn_candidate_scorer",
    },
    {
        "component": "candidate pool merge",
        "entrypoints": ["src/eval/candidate_pool_merge.py"],
        "classification": "ORACLE_NEUTRAL",
        "reuse_policy": "only_after_both_inputs_pass_gnn_provenance_gate",
    },
    {
        "component": "cross-parent verification and WNode matrix",
        "entrypoints": ["src/eval/wnode_action_matrix.py"],
        "classification": "RF_CONTAMINATED",
        "reuse_policy": "requires_batched_gnn_verification_adapter",
    },
    {
        "component": "prefix-aware selector",
        "entrypoints": ["src/eval/wnode_prefix_selector.py"],
        "classification": "ORACLE_NEUTRAL_ALGORITHM_RF_CONTAMINATED_ARTIFACTS",
        "reuse_policy": "only_after_new_gnn_matrix_schema",
    },
    {
        "component": "held-out BACE evaluator and final artifacts",
        "entrypoints": ["scripts/evaluate_bace_method.py"],
        "classification": "RF_CONTAMINATED",
        "reuse_policy": "requires_new_one_shot_gnn_heldout_evaluator",
    },
)


def _absolute(value: str | Path, *, label: str, must_exist: bool) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute: {candidate}")
    return candidate.resolve(strict=must_exist)


def _fresh_output(value: str | Path) -> Path:
    output = _absolute(value, label="output directory", must_exist=False)
    if output.exists():
        raise FileExistsError(f"Output directory must be absent: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return output


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No deletion records found: {path}")
    return rows


def _validate_bace_checkpoint(
    checkpoint: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    audit = verify_checkpoint_bundle(checkpoint, verify_hashes=verify_hashes)
    card = audit["model_card"]
    required = {
        "dataset": "bace",
        "num_classes": 2,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
    }
    failures = [
        f"{key}={card.get(key)!r}"
        for key, expected in required.items()
        if card.get(key) != expected
    ]
    if failures:
        raise ValueError("BACE frozen-GNN checkpoint contract failed: " + ", ".join(failures))
    temperature = _read_json(checkpoint / "temperature_scaling.json")
    if (
        temperature.get("status") != "fit"
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
        or temperature.get("argmax_invariant") is not True
    ):
        raise ValueError("The downstream route requires a validation-only calibrated B4 bundle")
    return card


def _provenance(
    checkpoint: Path,
    card: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": "bace_frozen_gnn_oracle_provenance_v1",
        "dataset": "bace",
        "stage": stage,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "backbone": card["backbone"],
        "num_classes": 2,
        "source_label": 1,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": card["checkpoint_id"],
        "checkpoint_sha256sums_sha256": sha256_file(checkpoint / "sha256sums.txt"),
        "probability_source": "validation_temperature_scaled_softmax",
        "strict_flip_contract": "pred_before_equals_source_and_pred_after_differs",
        "test_loaded": False,
    }


def _graph_from_smiles(
    featurizer: MolecularGraphFeaturizer,
    *,
    smiles: str,
    molecule_id: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=-1,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split="calibration_scoring_preflight",
        graph_sha256=features.graph_sha256,
    )


def run_scoring_preflight(args: argparse.Namespace) -> int:
    if args.max_records < 1 or args.max_records > 64:
        raise ValueError("B6 max-records must be in [1, 64]")
    if args.batch_size < 1:
        raise ValueError("B6 batch-size must be positive")
    checkpoint = _absolute(args.checkpoint_dir, label="B4 checkpoint", must_exist=True)
    b5_output = _absolute(args.oracle_smoke_dir, label="B5 output", must_exist=True)
    output = _fresh_output(args.output_dir)
    # B4's complete bundle is verified once at the executable B6 boundary.
    card = _validate_bace_checkpoint(checkpoint)
    b5_summary = _read_json(b5_output / "oracle_smoke.json")
    required_b5 = {
        "status": "PASS",
        "evaluation_split": "calibration",
        "test_loaded": False,
        "rf_guard_pass": True,
        "selected_count": 16,
        "checkpoint_id": card["checkpoint_id"],
    }
    b5_failures = [
        f"{key}={b5_summary.get(key)!r}"
        for key, expected in required_b5.items()
        if b5_summary.get(key) != expected
    ]
    if b5_failures:
        raise ValueError("B6 rejected the B5 oracle smoke: " + ", ".join(b5_failures))
    if Path(str(b5_summary.get("checkpoint_dir", ""))).resolve(strict=True) != checkpoint:
        raise ValueError("B5 checkpoint path differs from the supplied frozen B4 bundle")
    if b5_summary.get("checkpoint_sha256sums_sha256") != sha256_file(
        checkpoint / "sha256sums.txt"
    ):
        raise ValueError("B5 checkpoint manifest identity differs from B4")

    deletion_path = b5_output / "deletion_records.jsonl"
    rows = _read_jsonl(deletion_path, limit=args.max_records)
    schema = MolecularFeatureSchema.from_dict(
        _read_json(checkpoint / "feature_schema.json")
    )
    featurizer = MolecularGraphFeaturizer(schema)
    pair_graphs: list[MolecularGraphData] = []
    verified_match_counts: list[int] = []
    for index, row in enumerate(rows):
        if row.get("residual_connected") is not True or row.get("sanitize_ok") is not True:
            raise ValueError(f"B6 row {index} is not a connected sanitized deletion")
        if int(row.get("source_label", -1)) != 1:
            raise ValueError(f"B6 row {index} has the wrong source label")
        outcomes = enumerate_connected_hard_deletions(
            str(row["parent_smiles"]),
            str(row["fragment_smiles"]),
            parent_id=str(row["parent_id"]),
            candidate_id=f"b6-scoring-{index:04d}",
        )
        matching = [
            outcome
            for outcome in outcomes
            if outcome.valid
            and outcome.residual_smiles == str(row["residual_smiles"])
        ]
        if not matching:
            raise ValueError(
                f"B6 row {index} cannot reproduce its connected hard deletion"
            )
        verified_match_counts.append(len(matching))
        pair_graphs.extend(
            (
                _graph_from_smiles(
                    featurizer,
                    smiles=str(row["parent_smiles"]),
                    molecule_id=f"b6-parent-{index}",
                ),
                _graph_from_smiles(
                    featurizer,
                    smiles=str(row["residual_smiles"]),
                    molecule_id=f"b6-residual-{index}",
                ),
            )
        )

    # The factory loads and validates exactly one task-specific oracle.  All
    # parent/residual predictions below share that object and one batched call.
    oracle = build_oracle(
        dataset="bace",
        backend="gnn",
        checkpoint=checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    predictions = oracle.predict_records(pair_graphs, batch_size=args.batch_size)
    if len(predictions) != 2 * len(rows):
        raise RuntimeError("B6 batched oracle returned the wrong number of records")
    single_probe = oracle.predict_proba(pair_graphs[:2], batch_size=1)
    batch_probe = np.asarray(
        [prediction["probabilities"] for prediction in predictions[:2]],
        dtype=np.float64,
    )
    batch_single_max_abs_difference = float(np.max(np.abs(single_probe - batch_probe)))
    if not np.isfinite(batch_probe).all() or batch_single_max_abs_difference > 1e-7:
        raise RuntimeError("B6 batch/single calibrated GNN probabilities differ")

    scored: list[dict[str, Any]] = []
    input_probability_max_abs_difference = 0.0
    for index, row in enumerate(rows):
        before = predictions[2 * index]
        after = predictions[2 * index + 1]
        semantics = compute_counterfactual_semantics(
            source_label=1,
            pred_before=before["predicted_label"],
            pred_after=after["predicted_label"],
            probabilities_before=before["probabilities"],
            probabilities_after=after["probabilities"],
            rule_id=f"b6-scoring-{index:04d}",
        )
        if semantics.pred_before != 1:
            raise RuntimeError(f"B6 row {index} escaped the frozen source cohort")
        recorded_before = np.asarray(row["probabilities_before"], dtype=np.float64)
        recorded_after = np.asarray(row["probabilities_after"], dtype=np.float64)
        recomputed_before = np.asarray(before["probabilities"], dtype=np.float64)
        recomputed_after = np.asarray(after["probabilities"], dtype=np.float64)
        difference = float(
            max(
                np.max(np.abs(recorded_before - recomputed_before)),
                np.max(np.abs(recorded_after - recomputed_after)),
            )
        )
        input_probability_max_abs_difference = max(
            input_probability_max_abs_difference, difference
        )
        if difference > 1e-7:
            raise RuntimeError(f"B6 row {index} differs from frozen B5 probabilities")
        if bool(row.get("cf_flip")) != semantics.cf_flip or not math.isclose(
            float(row.get("cf_drop")), semantics.cf_drop, rel_tol=0.0, abs_tol=1e-7
        ):
            raise RuntimeError(f"B6 row {index} differs from frozen B5 CF semantics")

        # This is deliberately named a diagnostic score.  It proves that the
        # GNN-derived signal is finite and consumable without claiming a PPO
        # reward contract or an optimizer update.
        diagnostic_score = float(semantics.cf_drop + (1.0 if semantics.cf_flip else 0.0))
        if not math.isfinite(diagnostic_score):
            raise RuntimeError("B6 diagnostic score is non-finite")
        scored.append(
            {
                "dataset": "bace",
                "candidate_id": f"b6-scoring-{index:04d}",
                "source_label": 1,
                "parent_id": row["parent_id"],
                "parent_smiles": row["parent_smiles"],
                "raw_fragment": row["fragment_smiles"],
                "core_fragment": row["fragment_smiles"],
                "final_fragment": row["fragment_smiles"],
                "valid": True,
                "parse_ok": True,
                "direct_substructure": True,
                "projection_used": False,
                "deletion_valid": True,
                "verified_match_count": verified_match_counts[index],
                "residual_smiles": row["residual_smiles"],
                **semantics.to_dict(),
                "oracle_backend": "gnn",
                "classifier_type": "gnn",
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": card["checkpoint_id"],
                "ppo_checkpoint_hash": None,
                "diagnostic_score": diagnostic_score,
                "diagnostic_score_contract": "cf_drop_plus_strict_flip_probe_only",
                "ppo_reward": None,
            }
        )

    provenance = _provenance(checkpoint, card, stage=B6_STAGE)
    provenance.update(
        {
            "proposal_source": "b5_real_connected_deletion_probe",
            "ppo_reward_backend": "not_executed",
            "ppo_training_performed": False,
        }
    )
    summary = {
        "schema_version": "bace_b6_gnn_scoring_preflight_v1",
        "dataset": "bace",
        "stage": B6_STAGE,
        "status": "BLOCKED",
        "diagnostic_status": "PASS",
        "stage_gate_status": "BLOCKED",
        "blocker_code": "BLOCKED_MISSING_GNN_PPO_INTEGRATION",
        "secondary_blockers": [
            "BLOCKED_NO_GNN_CLEAN_BACE_POLICY_INITIALIZATION"
        ],
        "execution_mode": "gnn_scoring_preflight_not_ppo",
        "scientific_claim": "calibrated_gnn_scoring_preflight_only",
        "ppo_training_performed": False,
        "ppo_pass_claimed": False,
        "downstream_release_authorized": False,
        "next_stage_launch_allowed": False,
        "checkpoint_id": card["checkpoint_id"],
        "backbone": card["backbone"],
        "source_label": 1,
        "num_classes": 2,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "oracle_load_count": 1,
        "scored_record_count": len(scored),
        "strict_flip_count": sum(bool(row["cf_flip"]) for row in scored),
        "finite_diagnostic_scores": all(
            math.isfinite(float(row["diagnostic_score"])) for row in scored
        ),
        "batch_single_max_abs_difference": batch_single_max_abs_difference,
        "b5_probability_max_abs_difference": input_probability_max_abs_difference,
        "b5_output": str(b5_output),
        "b5_deletion_records": str(deletion_path),
        "test_loaded": False,
        "completed_at": utc_now(),
    }
    route_audit = {
        "schema_version": "bace_frozen_gnn_legacy_route_audit_v1",
        "dataset": "bace",
        "status": "COMPLETE",
        "components": list(LEGACY_ROUTE_AUDIT),
        "historical_artifacts_promotable": False,
        "chemllm_role": "proposal_only",
        "classifier_role": "task_specific_frozen_gnn",
        "distance_encoder_role": "independent_molclr",
        "selector_role": "calibration_only_after_gnn_verification",
        "audited_at": utc_now(),
    }
    blocker = {
        "schema_version": "bace_frozen_gnn_stage_blocker_v1",
        "dataset": "bace",
        "stage": B6_STAGE,
        "status": "BLOCKED",
        "blocker_code": "BLOCKED_MISSING_GNN_PPO_INTEGRATION",
        "secondary_blockers": [
            "BLOCKED_NO_GNN_CLEAN_BACE_POLICY_INITIALIZATION"
        ],
        "diagnostic_preflight_status": "PASS",
        "missing_interface": STAGE_BLOCKERS["B7_PPO_FULL"]["missing_interface"],
        "legacy_entrypoints_audited": STAGE_BLOCKERS["B7_PPO_FULL"][
            "legacy_entrypoints"
        ],
        "legacy_artifact_classification": "RF_CONTAMINATED",
        "legacy_reuse_allowed": False,
        "scored_candidate_evidence": "scored_candidates.jsonl",
        "ppo_training_performed": False,
        "ppo_update_count": 0,
        "stage_completion_claimed": False,
        "policy_initialization_audit": {
            "historical_bace_ppo": "RF_CONTAMINATED_DIAGNOSIS_ONLY",
            "unknown_provenance_lora": "FORBIDDEN",
            "chemllm_base": "ORACLE_NEUTRAL_BUT_CURRENT_PPO_ENTRY_REQUIRES_LORA",
            "safe_reusable_policy_found": False,
        },
        "test_loaded": False,
        "recorded_at": utc_now(),
    }
    requirements = {
        "schema_version": "bace_frozen_gnn_stage_requirements_v1",
        "stage": B6_STAGE,
        "required_to_pass": {
            "minimum_ppo_updates": 1,
            "reward_oracle_backend": "gnn",
            "reward_checkpoint_id": card["checkpoint_id"],
            "ppo_training_performed": True,
            "required_output": "ppo_smoke_manifest.json",
            "policy_initialization": (
                "BACE-specific GNN-clean SFT checkpoint or provenance-clean "
                "generic molecular SFT checkpoint accepted by the PPO entrypoint"
            ),
        },
        "policy_initialization_constraints": {
            "historical_bace_ppo": "RF_CONTAMINATED",
            "unknown_provenance_lora": "REJECT",
            "chemllm_base": "ORACLE_NEUTRAL_NOT_DIRECTLY_ACCEPTED_BY_CURRENT_PPO_ENTRY",
        },
        "current_preflight_satisfies_stage": False,
        "blocker_exit_code": SCIENTIFIC_BLOCKED_EXIT_CODE,
    }
    _write_jsonl(output / "scored_candidates.jsonl", scored)
    atomic_write_json(output / "oracle_provenance.json", provenance)
    atomic_write_json(output / "legacy_route_audit.json", route_audit)
    atomic_write_json(output / "stage_requirements.json", requirements)
    atomic_write_json(output / "blocker.json", blocker)
    atomic_write_json(output / "b6_scoring_preflight.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BACE_GNN_SCORING_PREFLIGHT_PASS_NOT_PPO]", flush=True)
    print("[BLOCKED_MISSING_GNN_PPO_INTEGRATION]", flush=True)
    print("[BACE_GNN_STAGE_BLOCKED]", flush=True)
    return SCIENTIFIC_BLOCKED_EXIT_CODE


def _validate_predecessor_output(stage: str, predecessor: Path) -> list[str]:
    required = PREDECESSOR_OUTPUT_CONTRACT[stage]
    missing = [name for name in required if not (predecessor / name).is_file()]
    if missing:
        raise ValueError(
            f"{stage} predecessor output is missing its scientific contract: {missing}"
        )
    checks: list[str] = [str((predecessor / name).resolve()) for name in required]
    if stage == "B7_PPO_FULL":
        b6 = _read_json(predecessor / "ppo_smoke_manifest.json")
        if (
            b6.get("status") != "PASS"
            or b6.get("ppo_training_performed") is not True
            or int(b6.get("ppo_update_count", 0)) < 1
            or b6.get("reward_oracle_backend") != "gnn"
        ):
            raise ValueError("B7 requires a real GNN-reward PPO smoke PASS")
    else:
        json_candidates = [
            predecessor / name for name in required if name.endswith(".json")
        ]
        if json_candidates:
            payload = _read_json(json_candidates[0])
            if payload.get("status") not in {"PASS", "FROZEN"}:
                raise ValueError(
                    f"{stage} predecessor scientific manifest is not PASS/FROZEN"
                )
    return checks


def run_stage_blocker(args: argparse.Namespace) -> int:
    stage = str(args.stage)
    if stage not in BLOCKED_STAGES:
        raise ValueError(f"No blocker contract exists for stage {stage}")
    checkpoint = _absolute(args.checkpoint_dir, label="B4 checkpoint", must_exist=True)
    predecessor = _absolute(
        args.predecessor_output, label="predecessor output", must_exist=True
    )
    output = _fresh_output(args.output_dir)
    # B4 and B6 already verified the complete bundle.  Blocked preflights check
    # its contract without repeatedly hashing every model artifact.
    card = _validate_bace_checkpoint(checkpoint, verify_hashes=False)
    checked_inputs = _validate_predecessor_output(stage, predecessor)
    spec = dict(STAGE_BLOCKERS[stage])
    blocker_code = str(spec["code"])
    secondary_blockers: list[str] = []

    if stage == "B10_POOL_MERGED":
        if args.base_pool_output is None:
            raise ValueError("B10 requires --base-pool-output in addition to B9")
        base_pool = _absolute(
            args.base_pool_output, label="B8 base-pool output", must_exist=True
        )
        for name in ("candidate_pool.jsonl", "pool_manifest.json"):
            path = base_pool / name
            if not path.is_file():
                raise ValueError(f"B10 base pool is missing {name}")
            checked_inputs.append(str(path.resolve()))

    if stage in {"B11_CROSS_PARENT_VERIFIED", "B13_FINAL_EVAL"}:
        if args.molclr_checkpoint is None:
            blocker_code = "MOLCLR_MISSING_BLOCKS_WNODE_FULL"
            secondary_blockers.append(str(spec["code"]))
        else:
            molclr = _absolute(
                args.molclr_checkpoint, label="MolCLR checkpoint", must_exist=True
            )
            if not molclr.is_file() or molclr.stat().st_size <= 0:
                raise ValueError("MolCLR checkpoint must be a non-empty file")
            checked_inputs.append(str(molclr))

    if stage == "B13_FINAL_EVAL":
        if args.test_csv is None:
            raise ValueError("B13 requires the frozen held-out --test-csv path")
        test_csv = _absolute(args.test_csv, label="held-out test CSV", must_exist=True)
        if not test_csv.is_file() or test_csv.stat().st_size <= 0:
            raise ValueError("B13 held-out test CSV is missing or empty")
        # A blocked preflight deliberately does not parse or featurize test rows.
        checked_inputs.append(str(test_csv))

    provenance = _provenance(checkpoint, card, stage=stage)
    blocker = {
        "schema_version": "bace_frozen_gnn_stage_blocker_v1",
        "dataset": "bace",
        "stage": stage,
        "status": "BLOCKED",
        "blocker_code": blocker_code,
        "secondary_blockers": secondary_blockers,
        "missing_interface": spec["missing_interface"],
        "legacy_entrypoints_audited": spec["legacy_entrypoints"],
        "legacy_artifact_classification": "RF_CONTAMINATED",
        "legacy_reuse_allowed": False,
        "legacy_reason": spec["legacy_reason"],
        "required_next_outputs": spec["required_next_outputs"],
        "predecessor_output": str(predecessor),
        "checked_inputs": checked_inputs,
        "oracle_checkpoint_id": card["checkpoint_id"],
        "test_loaded": False,
        "recorded_at": utc_now(),
    }
    requirements = {
        "schema_version": "bace_frozen_gnn_stage_requirements_v1",
        "stage": stage,
        "required_oracle_contract": {
            "dataset": "bace",
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "num_classes": 2,
            "source_label": 1,
        },
        "required_next_outputs": spec["required_next_outputs"],
        "forbidden_legacy_reuse": True,
        "blocker_exit_code": SCIENTIFIC_BLOCKED_EXIT_CODE,
    }
    atomic_write_json(output / "oracle_provenance.json", provenance)
    atomic_write_json(output / "stage_requirements.json", requirements)
    atomic_write_json(output / "blocker.json", blocker)
    print(json.dumps(blocker, sort_keys=True), flush=True)
    print(f"[{blocker_code}]", flush=True)
    print("[BACE_GNN_STAGE_BLOCKED]", flush=True)
    return SCIENTIFIC_BLOCKED_EXIT_CODE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)

    smoke = commands.add_parser("scoring-preflight")
    smoke.add_argument("--checkpoint-dir", required=True)
    smoke.add_argument("--oracle-smoke-dir", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--batch-size", type=int, default=32)
    smoke.add_argument("--max-records", type=int, default=32)

    blocker = commands.add_parser("stage-blocker")
    blocker.add_argument("--stage", choices=BLOCKED_STAGES, required=True)
    blocker.add_argument("--checkpoint-dir", required=True)
    blocker.add_argument("--predecessor-output", required=True)
    blocker.add_argument("--output-dir", required=True)
    blocker.add_argument("--base-pool-output")
    blocker.add_argument("--molclr-checkpoint")
    blocker.add_argument("--test-csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "scoring-preflight":
        return run_scoring_preflight(args)
    if args.action == "stage-blocker":
        return run_stage_blocker(args)
    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
