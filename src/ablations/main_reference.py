"""Build a hash-closed BACE/Ours reference from completed main-table artifacts.

The builder is intentionally evidence-driven: every value comes from an input
manifest or an artifact digest.  It never substitutes a checkpoint from a
different dataset and records an unavailable matched SFT lineage explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.ablations.contracts import (
    ContractError,
    canonical_json_sha256,
    require_sha256,
    sha256_file,
    validate_main_reference_contract,
)
from src.ablations.launch_gate import validate_matrix_authority_pointer


EXPECTED_PARENT_COUNT = 386
EXPECTED_BASE_SAMPLING = {
    "num_return_sequences": 4,
    "seed": 7,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_new_tokens": 96,
}
EXPECTED_HIGH_TEMPERATURE_SAMPLING = {
    "num_return_sequences": 4,
    "seed": 13,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 96,
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected JSON object: {path}")
    return payload


def _require_pass(payload: Mapping[str, Any], *, name: str) -> None:
    passed = (
        payload.get("status") == "PASS"
        or payload.get("passed") is True
        or payload.get("audit_passed") is True
    )
    if not passed:
        raise ContractError(f"{name} is not PASS")


def _require_frozen_selector(payload: Mapping[str, Any]) -> None:
    """Accept only the exact terminal state emitted by the B12 selector.

    The existing BACE/Ours selector predates the generic PASS convention.  Its
    scientific terminal is ``FROZEN`` and is safe only when calibration was
    loaded, fitting is explicitly calibration-only, the selection is frozen,
    and held-out test was not opened.  Treating arbitrary non-failure statuses
    as success would hide incomplete selectors, so keep this compatibility
    rule deliberately narrow.
    """

    required = {
        "status": "FROZEN",
        "stage": "B12_SELECTOR",
        "calibration_loaded": True,
        "selector_fitted_on_calibration": True,
        "selection_frozen": True,
        "test_loaded": False,
        "K": 20,
        "cf_mode": "strict_flip",
        "classifier_type": "gnn",
        "oracle_backend": "gnn",
        "source_label": 1,
        "num_classes": 2,
    }
    for field, expected in required.items():
        _same(payload.get(field), expected, field=f"selector {field}")
    ordered = payload.get("ordered_rule_ids")
    if not isinstance(ordered, list) or len(ordered) != 20:
        raise ContractError("selector ordered_rule_ids must contain the frozen top 20")
    if len(set(ordered)) != len(ordered) or any(
        not isinstance(rule_id, str) or not rule_id for rule_id in ordered
    ):
        raise ContractError("selector ordered_rule_ids are invalid or duplicated")
    require_sha256(
        payload.get("ordered_rule_ids_sha256"),
        field="selector.ordered_rule_ids_sha256",
    )
    require_sha256(
        payload.get("calibration_input_hash"),
        field="selector.calibration_input_hash",
    )


def _same(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise ContractError(f"{field} mismatch: {actual!r} != {expected!r}")


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing required artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


@dataclass(frozen=True, slots=True)
class BaceOursReferenceInputs:
    matrix_authority_state: Path
    final_root: Path
    oracle_root: Path
    ppo_root: Path
    train_parent_prep_manifest: Path
    base_pool_manifest: Path
    high_temperature_pool_manifest: Path
    merged_pool_manifest: Path
    verification_manifest: Path
    selector_manifest: Path
    molclr_checkpoint: Path


def build_bace_ours_main_reference(
    inputs: BaceOursReferenceInputs,
) -> dict[str, Any]:
    """Return the frozen main-reference contract or fail closed."""

    authority_state = _load(inputs.matrix_authority_state)
    authority = validate_matrix_authority_pointer(authority_state)
    # Freezing the already-published BACE/Ours cell is framework construction,
    # not ablation science.  It is intentionally allowed while the remaining
    # main-table cells are still running; the separate launch gate remains
    # strict 16/16.  The cell must nevertheless come from the independently
    # reopened current authority, never from a remembered or arbitrary root.
    if "BACE/Ours" not in set(authority.get("applied_cells", ())):
        raise ContractError("current matrix authority has not published BACE/Ours")
    final_root = inputs.final_root.expanduser().resolve(strict=True)
    authority_bace_ours = Path(
        str(authority.get("cell_roots", {}).get("BACE/Ours") or "")
    ).expanduser()
    if (
        not authority_bace_ours.is_absolute()
        or authority_bace_ours.resolve(strict=False) != final_root
    ):
        raise ContractError("BACE/Ours final root is not the unique matrix authority cell")

    final_run = _load(final_root / "run_manifest.json")
    final_eval = _load(final_root / "evaluation_manifest.json")
    final_summary = _load(final_root / "summary.json")
    final_audit = _load(final_root / "final_artifact_audit.json")
    split_manifest = _load(inputs.oracle_root / "split_manifest.json")
    model_card = _load(inputs.oracle_root / "model_card.json")
    temperature = _load(inputs.oracle_root / "temperature_scaling.json")
    policy_manifest = _load(inputs.ppo_root / "ppo_manifest.json")
    policy_run = _load(inputs.ppo_root / "run_manifest.json")
    policy_provenance = _load(inputs.ppo_root / "policy_provenance.json")
    prep = _load(inputs.train_parent_prep_manifest)
    base_pool = _load(inputs.base_pool_manifest)
    high_pool = _load(inputs.high_temperature_pool_manifest)
    merged_pool = _load(inputs.merged_pool_manifest)
    verification = _load(inputs.verification_manifest)
    selector = _load(inputs.selector_manifest)

    for name, payload in (
        ("final run", final_run),
        ("final evaluation", final_eval),
        ("final audit", final_audit),
        ("policy", policy_manifest),
        ("parent preparation", prep),
        ("base candidate pool", base_pool),
        ("high-temperature candidate pool", high_pool),
        ("merged candidate pool", merged_pool),
        ("verification", verification),
    ):
        _require_pass(payload, name=name)
    _require_frozen_selector(selector)

    _same(str(final_run.get("dataset", "")).lower(), "bace", field="dataset")
    _same(str(final_run.get("method", "")).lower(), "ours", field="method")
    _same(final_run.get("rf_oracle_used"), False, field="rf_oracle_used")
    _same(final_run.get("cf_mode"), "strict_flip", field="cf_mode")
    _same(final_run.get("source_label"), 1, field="source_label")
    _same(final_eval.get("test_used_for_selection"), False, field="test selection")
    _same(final_eval.get("selection_frozen_before_test"), True, field="freeze order")

    model = _file_identity(inputs.oracle_root / "model.pt")
    model_schema = _file_identity(inputs.oracle_root / "feature_schema.json")
    temperature_file = _file_identity(inputs.oracle_root / "temperature_scaling.json")
    split_file = _file_identity(inputs.oracle_root / "split_manifest.json")
    molclr = _file_identity(inputs.molclr_checkpoint)
    _same(model["sha256"], model_card.get("checkpoint_id"), field="GINE checkpoint")
    _same(
        model["sha256"], final_run.get("oracle_checkpoint_hash"), field="final oracle"
    )
    _same(
        temperature_file["sha256"],
        final_eval.get("temperature_sha256", temperature_file["sha256"]),
        field="temperature artifact",
    )
    _same(
        molclr["sha256"], final_run.get("molclr_checkpoint_hash"), field="MolCLR"
    )

    split_files = split_manifest.get("files")
    if not isinstance(split_files, Mapping):
        raise ContractError("oracle split manifest has no files mapping")
    split_hashes: dict[str, str] = {}
    split_paths: dict[str, str] = {}
    for split in ("train", "validation", "calibration", "test"):
        identity = split_files.get(split)
        if not isinstance(identity, Mapping):
            raise ContractError(f"missing split identity: {split}")
        split_path = Path(str(identity.get("path", "")))
        expected_sha = str(identity.get("sha256", ""))
        if not split_path.is_file():
            raise ContractError(f"missing split file: {split_path}")
        _same(sha256_file(split_path), expected_sha, field=f"{split} split")
        split_hashes[split] = expected_sha
        split_paths[split] = str(split_path)

    policy_weights = _file_identity(inputs.ppo_root / "adapter_model.safetensors")
    policy_config = _file_identity(inputs.ppo_root / "adapter_config.json")
    policy_hash = str(policy_manifest.get("policy_checkpoint_hash", ""))
    require_sha256(policy_hash, field="policy_checkpoint_hash")
    _same(base_pool.get("policy_checkpoint_hash"), policy_hash, field="base policy")
    _same(high_pool.get("policy_checkpoint_hash"), policy_hash, field="high-temp policy")
    _same(merged_pool.get("policy_checkpoint_hash"), policy_hash, field="merged policy")
    _same(selector.get("policy_checkpoint_hash"), policy_hash, field="selector policy")
    candidate_universe_sha = require_sha256(
        merged_pool.get("candidate_universe_hash"),
        field="merged_pool.candidate_universe_hash",
    )
    # B10 records both a serialized-pool hash and the canonical scientific
    # candidate-universe hash.  B11/B12 intentionally bind the latter.
    _same(
        verification.get("candidate_source_hash"),
        candidate_universe_sha,
        field="verification candidate universe",
    )
    _same(
        selector.get("candidate_pool_hash"),
        candidate_universe_sha,
        field="selector candidate universe",
    )
    _same(
        selector.get("oracle_checkpoint_hash"),
        model["sha256"],
        field="selector oracle",
    )
    _same(
        selector.get("molclr_checkpoint_hash"),
        molclr["sha256"],
        field="selector MolCLR",
    )
    selector_matrix_identity = selector.get("matrix_manifest_identity")
    if not isinstance(selector_matrix_identity, Mapping):
        raise ContractError("selector has no B11 matrix manifest identity")
    selector_matrix_path = Path(str(selector_matrix_identity.get("path") or ""))
    selector_matrix = _file_identity(selector_matrix_path)
    _same(
        selector_matrix["sha256"],
        selector_matrix_identity.get("sha256"),
        field="selector B11 matrix SHA",
    )
    matrix_manifest = _load(selector_matrix_path)
    _require_pass(matrix_manifest, name="B11 merged matrix")
    for field, expected in {
        "stage": "B11_CROSS_PARENT_VERIFIED",
        "calibration_loaded": True,
        "test_loaded": False,
        "candidate_universe_hash": candidate_universe_sha,
        "policy_checkpoint_hash": policy_hash,
        "oracle_checkpoint_hash": model["sha256"],
        "molclr_checkpoint_hash": molclr["sha256"],
    }.items():
        _same(matrix_manifest.get(field), expected, field=f"B11 matrix {field}")
    policy_identity = base_pool.get("policy_identity", {})
    _same(
        policy_identity.get("adapter_weights", {}).get("sha256"),
        policy_weights["sha256"],
        field="policy weights",
    )
    _same(
        policy_identity.get("adapter_config", {}).get("sha256"),
        policy_config["sha256"],
        field="policy config",
    )

    if policy_provenance.get("adapter_initialized_from_scratch") is not True:
        raise ContractError("BACE policy initializer provenance is ambiguous")
    if policy_provenance.get("data_split_used") != "none":
        raise ContractError("clean BACE policy initializer unexpectedly used data")
    if policy_provenance.get("matched_sft_checkpoint") not in (None, ""):
        raise ContractError("BACE provenance unexpectedly claims a matched SFT checkpoint")
    if policy_provenance.get("training_data_hash") not in (None, ""):
        raise ContractError("BACE clean initializer unexpectedly claims SFT training data")
    matched_sft_available = False

    train_prep = prep.get("fixed_shards", {}).get("train", {})
    _same(train_prep.get("parent_count"), EXPECTED_PARENT_COUNT, field="train parent count")
    _same(base_pool.get("all_parent_ids_sha256"), train_prep.get("parent_ids_sha256"), field="base cohort")
    _same(high_pool.get("all_parent_ids_sha256"), train_prep.get("parent_ids_sha256"), field="high-temp cohort")
    base_generation = dict(base_pool.get("generation_config", {}))
    high_generation = dict(high_pool.get("generation_config", {}))
    base_count = int(base_generation.get("num_return_sequences", 0))
    high_count = int(high_generation.get("num_return_sequences", 0))
    for field, expected in EXPECTED_BASE_SAMPLING.items():
        _same(base_generation.get(field), expected, field=f"base sampling {field}")
    for field, expected in EXPECTED_HIGH_TEMPERATURE_SAMPLING.items():
        _same(high_generation.get(field), expected, field=f"high-temp sampling {field}")

    policy_args = policy_run.get("args")
    if not isinstance(policy_args, Mapping):
        raise ContractError("PPO run manifest has no args mapping")
    projection_policy = {
        key: policy_args.get(key)
        for key in sorted(policy_args)
        if key.startswith("projection_") or key == "enable_parent_projection"
    }
    syntax_repair_policy = {
        key: policy_args.get(key)
        for key in sorted(policy_args)
        if key.startswith("repair_") or key == "enable_minimal_syntax_repair"
    }
    component_salvage_policy = {
        key: policy_args.get(key)
        for key in (
            "enable_component_salvage",
            "component_salvage_method",
            "component_salvage_min_atoms",
        )
    }

    wnode_config = {
        "distance_line": verification.get("distance_provider_stats", {}).get("distance_line"),
        "distance_namespace": verification.get("distance_provider_stats", {}).get("distance_namespace"),
        "distance_type": verification.get("distance_type"),
        "feature_cost": verification.get("distance_provider_stats", {}).get("feature_cost"),
        "node_mass": verification.get("distance_provider_stats", {}).get("node_mass"),
        "size_penalty_beta": verification.get("distance_provider_stats", {}).get("size_penalty_beta"),
        "solver": verification.get("distance_provider_stats", {}).get("solver"),
        "match_selection_policy": verification.get("match_selection_policy"),
        "no_valid_strict_flip_semantics": verification.get("no_valid_strict_flip_semantics"),
    }
    if any(value is None for value in wnode_config.values()):
        raise ContractError("incomplete WNode configuration in verification manifest")

    selector_identity = _file_identity(inputs.selector_manifest)
    thresholds_path = inputs.selector_manifest.parent / "thresholds.json"
    variant_configs_path = inputs.selector_manifest.parent / "variant_configs.json"
    thresholds_identity = _file_identity(thresholds_path)
    variant_configs_identity = _file_identity(variant_configs_path)
    final_evaluation_identity = _file_identity(final_root / "evaluation_manifest.json")

    blocked_sft = {
        "status": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
        "reason": (
            "BACE main policy provenance declares chemllm_base_fresh_lora and "
            "adapter_initialized_from_scratch=true; no independently matched SFT artifact exists"
        ),
    }
    availability: dict[str, Any] = {
        "BRICS_FIXED": {"status": "CONFIG_ONLY", "checkpoint": None},
        "CHEMLLM_PRETRAINED": {
            "status": "AVAILABLE",
            "checkpoint": policy_provenance.get("source_model_path"),
            "checkpoint_sha": policy_provenance.get("source_model_hash"),
        },
        "CHEMLLM_SFT": blocked_sft,
        "CHEMLLM_SFT_PPO": blocked_sft,
    }

    proposal_contract = {
        "main_proposer_lineage": "CHEMLLM_BASE_FRESH_LORA_PPO",
        "base_model_path": policy_provenance.get("source_model_path"),
        "base_model_sha": policy_provenance.get("source_model_hash"),
        "tokenizer": _file_identity(inputs.ppo_root / "tokenizer.model"),
        "clean_policy_initializer": policy_provenance.get("adapter_dir"),
        "clean_policy_initializer_sha": policy_provenance.get("policy_initializer_hash"),
        "matched_sft_checkpoint": None,
        "ppo_checkpoint": str(inputs.ppo_root),
        "ppo_checkpoint_sha": policy_hash,
        "ppo_adapter_weights": policy_weights,
        "ppo_adapter_config": policy_config,
        "proposal_parent_cohort": train_prep.get("parent_manifest_identity"),
        "proposal_parent_count": train_prep.get("parent_count"),
        "candidate_attempts_per_parent": base_count + high_count,
        "base_sampling_count": base_count,
        "high_temperature_sampling_count": high_count,
        "base_sampling": base_generation,
        "high_temperature_sampling": high_generation,
        "prompt_schema": {
            "include_label_in_prompt": policy_args.get("include_label_in_prompt"),
            "core_output_mode": policy_args.get("core_output_mode"),
            "source_execution_commit": policy_run.get("git_commit"),
        },
        "projection_policy": projection_policy,
        "syntax_repair_policy": syntax_repair_policy,
        "component_salvage_policy": component_salvage_policy,
        "candidate_merge_dedup_policy": {
            "deterministic_merge": merged_pool.get("deterministic_merge"),
            "input_row_count": merged_pool.get("input_row_count"),
            "merged_row_count": merged_pool.get("merged_row_count"),
            "candidate_universe_count": merged_pool.get("candidate_universe_count"),
            "candidate_universe_sha": candidate_universe_sha,
            "pool_sha": merged_pool.get("candidate_pool_hash"),
        },
    }
    selector_contract = {
        "selector_manifest": selector_identity,
        "verified_matrix_manifest": selector_matrix,
        "thresholds": thresholds_identity,
        "variant_configs": variant_configs_identity,
        "K": selector.get("K"),
        "Table2_K": final_summary.get("table2_k"),
        "ordered_rule_ids_sha": selector.get("ordered_rule_ids_sha256"),
        "calibration_input_sha": selector.get("calibration_input_hash"),
        "threshold_config_sha": final_eval.get("threshold_config_hash"),
        "test_used_for_selection": final_eval.get("test_used_for_selection"),
    }

    payload: dict[str, Any] = {
        "schema_version": "bace_ours_main_reference_v1",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": final_run.get("source_label"),
        "main_final_root": str(final_root),
        "main_final_audit_sha": sha256_file(final_root / "final_artifact_audit.json"),
        "matrix_authority": authority,
        "main_execution_commit": policy_run.get("git_commit"),
        "gine_checkpoint": str(inputs.oracle_root / "model.pt"),
        "gine_checkpoint_root": str(inputs.oracle_root),
        "gine_checkpoint_sha": model["sha256"],
        "gine_training_commit": model_card.get("training_commit"),
        "temperature": temperature.get("temperature"),
        "temperature_sha": temperature_file["sha256"],
        "feature_schema": str(inputs.oracle_root / "feature_schema.json"),
        "feature_schema_sha": model_schema["sha256"],
        "molclr_root": str(inputs.molclr_checkpoint),
        "molclr_sha": molclr["sha256"],
        "wnode_config": wnode_config,
        "wnode_config_sha": canonical_json_sha256(wnode_config),
        "dataset_split_manifest": {"path": str(inputs.oracle_root / "split_manifest.json"), **split_file},
        "dataset_split_hashes": split_hashes,
        "dataset_split_paths": split_paths,
        "proposal_contract": proposal_contract,
        "selector_contract": selector_contract,
        "selector_config_sha": selector_identity["sha256"],
        "threshold_config_sha": str(final_eval.get("threshold_config_hash")),
        "evaluation_config_sha": final_evaluation_identity["sha256"],
        "llm_variant_availability": availability,
        "matched_sft_checkpoint_available": matched_sft_available,
        "scientific_values_inferred": False,
    }
    return validate_main_reference_contract(payload)


__all__ = ["BaceOursReferenceInputs", "build_bace_ours_main_reference"]
