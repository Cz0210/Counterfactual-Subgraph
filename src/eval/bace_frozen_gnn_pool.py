"""GNN-clean BACE candidate generation shards and deterministic B10 merge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.chem.hard_deletion import enumerate_connected_hard_deletions
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeaturizer,
)
from src.data.ppo_prompt_dataset import PPOPromptRecord
from src.eval.bace_frozen_gnn_contracts import (
    BACEParent,
    CLASSIFIER_TYPE,
    DATASET,
    NUM_CLASSES,
    NUM_SHARDS,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    assert_no_rf_provenance,
    assert_stage_data_boundary,
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fixed_parent_shard_map,
    fresh_output_dir,
    load_bace_parents,
    read_json,
    read_jsonl,
    select_parent_shard,
    sha256_file,
    stable_sha256,
    utc_now,
    validate_materialized_parent_shard,
    validate_pass_manifest,
)
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.eval.full_candidate_pool import (
    CONNECTED_DELETION_PROMPT_MODE,
    FullPoolGenerationConfig,
    _build_lora_model,
    _build_tokenizer,
    build_generation_kwargs,
    generate_ids_with_sanitized_kwargs,
    render_generation_prompt,
    resolve_adapter_load_path,
    set_global_generation_seed,
)
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.models.llm_generator import clean_generated_smiles
from src.oracles.gnn_oracle import verify_checkpoint_bundle
from src.oracles.oracle_factory import build_oracle


POOL_STAGES = ("B8_POOL_BASE", "B9_POOL_HIGHTEMP")


@dataclass(frozen=True, slots=True)
class BACEPoolConfig:
    stage: str
    temperature: float
    top_p: float
    num_return_sequences: int
    seed: int
    max_new_tokens: int = 96
    batch_size: int = 1
    oracle_batch_size: int = 256

    @classmethod
    def preregistered(cls, stage: str, *, batch_size: int = 1) -> "BACEPoolConfig":
        normalized = str(stage).strip().upper()
        if normalized == "B8_POOL_BASE":
            return cls(normalized, 0.30, 0.90, 4, 7, batch_size=int(batch_size))
        if normalized == "B9_POOL_HIGHTEMP":
            return cls(normalized, 0.70, 0.90, 4, 13, batch_size=int(batch_size))
        raise ValueError(f"Unsupported BACE candidate-pool stage: {stage}")


def _graph(
    featurizer: MolecularGraphFeaturizer,
    *,
    smiles: str,
    molecule_id: str,
    split: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=SOURCE_LABEL,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split=split,
        graph_sha256=features.graph_sha256,
    )


def _parent_prompt(parent: BACEParent) -> PPOPromptRecord:
    return PPOPromptRecord(
        parent_index=parent.source_row_index,
        parent_smiles=parent.smiles,
        label=parent.label,
        prompt=parent.prompt or "",
        raw_payload={"parent_id": parent.parent_id},
    )


def _validate_b7(
    b7_output: Path, *, policy_checkpoint: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_pass_manifest(
        b7_output / "ppo_manifest.json",
        expected_stage="B7_PPO_FULL",
        require_no_test=True,
    )
    oracle_provenance = read_json(b7_output / "oracle_provenance.json")
    assert_no_rf_provenance(oracle_provenance)
    if (
        manifest.get("ppo_training_performed") is not True
        or int(manifest.get("ppo_update_count", 0)) < 1
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
    ):
        raise ValueError("B8/B9 require a passing train-only real B7 PPO run")
    resolved = resolve_adapter_load_path(policy_checkpoint)
    expected_hash = str(manifest.get("policy_checkpoint_hash") or "").strip()
    if not expected_hash:
        raise ValueError("B7 manifest lacks the frozen on-disk policy_checkpoint_hash")
    config_identity = manifest.get("final_adapter_config_identity")
    weights_identity = manifest.get("final_adapter_weights_identity")
    if not isinstance(config_identity, Mapping) or not isinstance(
        weights_identity, Mapping
    ):
        raise ValueError("B7 manifest lacks final adapter config/weights identities")
    weights_path = resolved / (
        "adapter_model.safetensors"
        if (resolved / "adapter_model.safetensors").is_file()
        else "adapter_model.bin"
    )
    current_config = file_identity(resolved / "adapter_config.json")
    current_weights = file_identity(weights_path)
    for role, declared, current in (
        ("config", config_identity, current_config),
        ("weights", weights_identity, current_weights),
    ):
        if dict(declared) != current:
            raise ValueError(
                f"B7 final adapter {role} identity differs from current file bytes"
            )
    hash_schema = manifest.get("policy_checkpoint_hash_schema")
    hash_payload = manifest.get("policy_checkpoint_hash_payload")
    if hash_schema != "bace_lora_checkpoint_identity_v1" or not isinstance(
        hash_payload, Mapping
    ):
        raise ValueError("B7 policy checkpoint hash schema/payload is missing")
    expected_payload = {
        "schema_version": "bace_lora_checkpoint_identity_v1",
        "adapter_config_name": Path(current_config["path"]).name,
        "adapter_config_sha256": current_config["sha256"],
        "adapter_config_size": current_config["size"],
        "adapter_weights_name": Path(current_weights["path"]).name,
        "adapter_weights_sha256": current_weights["sha256"],
        "adapter_weights_size": current_weights["size"],
    }
    if dict(hash_payload) != expected_payload:
        raise ValueError("B7 policy checkpoint hash payload differs from adapter bytes")
    recomputed_policy_hash = stable_sha256(expected_payload)
    if recomputed_policy_hash != expected_hash:
        raise ValueError("B7 policy_checkpoint_hash does not bind current adapter bytes")
    return manifest, {
        "resolved_adapter_path": str(resolved),
        "declared_policy_hash": expected_hash,
        "adapter_config": current_config,
        "adapter_weights": current_weights,
    }


def _checkpoint_contract(checkpoint: Path) -> tuple[dict[str, Any], MolecularFeatureSchema]:
    audit = verify_checkpoint_bundle(checkpoint, verify_hashes=False)
    card = dict(audit["model_card"])
    required = {
        "dataset": DATASET,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
    }
    failures = [
        f"{key}={card.get(key)!r}"
        for key, expected in required.items()
        if card.get(key) != expected
    ]
    if failures:
        raise ValueError("Frozen BACE checkpoint contract failed: " + ", ".join(failures))
    schema = MolecularFeatureSchema.from_dict(
        read_json(checkpoint / "feature_schema.json")
    )
    return card, schema


def _generated_candidates(
    parents: Sequence[BACEParent],
    *,
    base_model_path: Path,
    adapter_path: Path,
    config: BACEPoolConfig,
) -> list[tuple[BACEParent, int, str, str]]:
    import torch

    set_global_generation_seed(config.seed)
    tokenizer = _build_tokenizer(
        base_model_path=base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = _build_lora_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_device = next(model.parameters()).device
    generation_config = FullPoolGenerationConfig(
        prompt_mode=CONNECTED_DELETION_PROMPT_MODE,
        num_return_sequences=config.num_return_sequences,
        generation_temperature=config.temperature,
        generation_top_p=config.top_p,
        generation_do_sample=True,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.batch_size,
        seed=config.seed,
        enable_parent_projection=False,
        enable_projected_cf_reward=False,
        enable_substructure_distance_reward=False,
    )
    outputs: list[tuple[BACEParent, int, str, str]] = []
    for start in range(0, len(parents), config.batch_size):
        batch = list(parents[start : start + config.batch_size])
        prompts = [
            render_generation_prompt(
                _parent_prompt(parent), prompt_mode=CONNECTED_DELETION_PROMPT_MODE
            )
            for parent in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        kwargs = build_generation_kwargs(
            encoded=encoded,
            tokenizer=tokenizer,
            config=generation_config,
        )
        generated_ids = generate_ids_with_sanitized_kwargs(
            model, kwargs, torch_module=torch
        )
        response_ids = generated_ids[:, encoded["input_ids"].shape[1] :]
        raw_outputs = tokenizer.batch_decode(
            response_ids.detach().cpu().tolist(), skip_special_tokens=True
        )
        expected = len(batch) * config.num_return_sequences
        if len(raw_outputs) != expected:
            raise RuntimeError(
                f"ChemLLM generated {len(raw_outputs)} sequences, expected {expected}"
            )
        cursor = 0
        for parent in batch:
            for candidate_index in range(config.num_return_sequences):
                raw = str(raw_outputs[cursor])
                outputs.append(
                    (parent, candidate_index, raw, clean_generated_smiles(raw))
                )
                cursor += 1
    return outputs


def _score_generated_candidates(
    generated: Sequence[tuple[BACEParent, int, str, str]],
    *,
    oracle: Any,
    featurizer: MolecularGraphFeaturizer,
    stage: str,
    shard_index: int,
    oracle_batch_size: int,
    checkpoint_id: str,
) -> list[dict[str, Any]]:
    parent_order: list[BACEParent] = []
    seen_parent_ids: set[str] = set()
    for parent, _candidate_index, _raw, _fragment in generated:
        if parent.parent_id not in seen_parent_ids:
            parent_order.append(parent)
            seen_parent_ids.add(parent.parent_id)
    parent_graphs = [
        _graph(
            featurizer,
            smiles=parent.smiles,
            molecule_id=parent.parent_id,
            split="train",
        )
        for parent in parent_order
    ]
    parent_predictions = oracle.predict_records(
        parent_graphs, batch_size=oracle_batch_size
    )
    before_by_parent = dict(
        zip((parent.parent_id for parent in parent_order), parent_predictions, strict=True)
    )

    candidates: list[dict[str, Any]] = []
    valid_outcome_rows: list[tuple[int, Any]] = []
    residual_graphs: list[MolecularGraphData] = []
    for row_index, (parent, candidate_index, raw, fragment) in enumerate(generated):
        canonical_fragment = canonicalize_smiles(fragment) if fragment else None
        candidate_id = "BACEGEN_" + stable_sha256(
            {
                "stage": stage,
                "parent_id": parent.parent_id,
                "candidate_index": candidate_index,
                "raw": raw,
            }
        )[:24].upper()
        outcomes = enumerate_connected_hard_deletions(
            parent.smiles,
            canonical_fragment or fragment,
            parent_id=parent.parent_id,
            candidate_id=candidate_id,
        )
        row = {
            "dataset": DATASET,
            "stage": stage,
            "shard_index": int(shard_index),
            "candidate_id": candidate_id,
            "candidate_index": int(candidate_index),
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "label": SOURCE_LABEL,
            "source_label": SOURCE_LABEL,
            "raw_output": raw,
            "raw_fragment": fragment or None,
            "core_fragment": canonical_fragment,
            "final_fragment": canonical_fragment,
            "parse_ok": canonical_fragment is not None,
            "valid": canonical_fragment is not None,
            "connected": canonical_fragment is not None and "." not in canonical_fragment,
            "direct_substructure": bool(outcomes),
            "final_substructure": bool(outcomes),
            "projection_used": False,
            "projection_method": "none",
            "deletion_valid": any(outcome.valid for outcome in outcomes),
            "num_matches": len(outcomes),
            "num_valid_matches": sum(outcome.valid for outcome in outcomes),
            "oracle_ok": False,
            "cf_flip": False,
            "cf_drop": None,
            "reward_total": None,
            "pred_before": before_by_parent[parent.parent_id]["predicted_label"],
            "pred_after": None,
            "p_before": before_by_parent[parent.parent_id]["probabilities"],
            "p_after": None,
            "residual_smiles": None,
            "selected_match_index": None,
            "oracle_backend": ORACLE_BACKEND,
            "classifier_type": CLASSIFIER_TYPE,
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": checkpoint_id,
            "test_loaded": False,
            "calibration_loaded": False,
        }
        candidates.append(row)
        for outcome in outcomes:
            if not outcome.valid or not outcome.residual_smiles:
                continue
            valid_outcome_rows.append((row_index, outcome))
            residual_graphs.append(
                _graph(
                    featurizer,
                    smiles=outcome.residual_smiles,
                    molecule_id=f"{candidate_id}:match-{outcome.match_id}",
                    split="train_generated_residual",
                )
            )
    residual_predictions = (
        oracle.predict_records(residual_graphs, batch_size=oracle_batch_size)
        if residual_graphs
        else []
    )
    scored_by_row: dict[int, list[tuple[Any, dict[str, Any], Any]]] = {}
    for (row_index, outcome), after in zip(
        valid_outcome_rows, residual_predictions, strict=True
    ):
        before = before_by_parent[candidates[row_index]["parent_id"]]
        semantics = compute_counterfactual_semantics(
            source_label=SOURCE_LABEL,
            pred_before=before["predicted_label"],
            pred_after=after["predicted_label"],
            probabilities_before=before["probabilities"],
            probabilities_after=after["probabilities"],
            rule_id=candidates[row_index]["candidate_id"],
        )
        scored_by_row.setdefault(row_index, []).append((semantics, after, outcome))
    for row_index, values in scored_by_row.items():
        values.sort(
            key=lambda value: (
                -int(value[0].cf_flip),
                -float(value[0].cf_drop),
                int(value[2].match_id),
            )
        )
        semantics, after, outcome = values[0]
        reward = float(semantics.cf_drop + (1.0 if semantics.cf_flip else 0.0))
        if not math.isfinite(reward):
            raise RuntimeError("GNN candidate score produced a non-finite reward")
        candidates[row_index].update(
            {
                "oracle_ok": True,
                "cf_flip": semantics.cf_flip,
                "cf_drop": semantics.cf_drop,
                "reward_total": reward,
                "pred_after": after["predicted_label"],
                "p_after": after["probabilities"],
                "residual_smiles": outcome.residual_smiles,
                "selected_match_index": int(outcome.match_id),
            }
        )
    return candidates


def run_pool_shard(
    *,
    stage: str,
    train_split: str | Path,
    b7_output: str | Path,
    policy_checkpoint: str | Path,
    base_model_path: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    shard_index: int,
    device: str = "cuda:0",
    batch_size: int = 1,
    oracle_batch_size: int = 256,
    resume: bool = False,
    parent_shard_manifest: str | Path | None = None,
) -> dict[str, Any]:
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in POOL_STAGES:
        raise ValueError(f"Pool shard stage must be one of {POOL_STAGES}")
    assert_stage_data_boundary(stage=normalized_stage, split_path=train_split)
    train_path = Path(train_split).expanduser().resolve(strict=True)
    b7_root = Path(b7_output).expanduser().resolve(strict=True)
    policy_root = Path(policy_checkpoint).expanduser().resolve(strict=True)
    base_model = Path(base_model_path).expanduser().resolve(strict=True)
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    config = BACEPoolConfig.preregistered(normalized_stage, batch_size=batch_size)
    config = BACEPoolConfig(
        **{
            **asdict(config),
            "oracle_batch_size": int(oracle_batch_size),
        }
    )
    b7_manifest, policy_identity = _validate_b7(
        b7_root, policy_checkpoint=policy_root
    )
    adapter_path = Path(policy_identity["resolved_adapter_path"])
    card, schema = _checkpoint_contract(checkpoint)
    b7_oracle_id = b7_manifest.get("oracle_checkpoint_hash") or b7_manifest.get(
        "checkpoint_id"
    )
    if b7_oracle_id != card["checkpoint_id"]:
        raise ValueError("B7 and B8/B9 use different frozen GNN identities")
    all_parents = load_bace_parents(train_path)
    parents = select_parent_shard(all_parents, int(shard_index))
    parent_shard_identity = None
    if parent_shard_manifest is not None:
        validate_materialized_parent_shard(
            parent_shard_manifest,
            parents=all_parents,
            shard_index=int(shard_index),
            split="train",
        )
        parent_shard_identity = file_identity(parent_shard_manifest)
    if not parents:
        raise ValueError(f"Train shard {shard_index} is empty")
    parent_ids = [parent.parent_id for parent in parents]
    execution_fingerprint = stable_sha256(
        {
            "stage": normalized_stage,
            "shard_index": int(shard_index),
            "parent_ids": parent_ids,
            "train_split": file_identity(train_path),
            "policy_checkpoint_hash": policy_identity["declared_policy_hash"],
            "oracle_checkpoint_hash": card["checkpoint_id"],
            "generation_config": asdict(config),
            "parent_shard_manifest": parent_shard_identity,
        }
    )
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        raise ValueError(f"Pool shard output must be absolute: {output}")
    output = output.resolve(strict=False)
    if output.exists():
        if not resume:
            raise FileExistsError(f"Pool shard output already exists: {output}")
        progress = read_json(output / "IN_PROGRESS.json")
        if progress.get("execution_fingerprint") != execution_fingerprint:
            raise ValueError("Pool shard resume fingerprint differs from current inputs")
        if (output / "PASS").is_file():
            return validate_pass_manifest(
                output / "pool_manifest.json",
                expected_stage=normalized_stage,
                require_no_test=True,
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        atomic_json(
            output / "IN_PROGRESS.json",
            {
                "schema_version": "bace_frozen_gnn_pool_resume_v1",
                "status": "RUNNING",
                "stage": normalized_stage,
                "shard_index": int(shard_index),
                "execution_fingerprint": execution_fingerprint,
                "created_at": utc_now(),
            },
        )
    expected_rows = len(parents) * config.num_return_sequences
    if (output / "candidate_pool.jsonl").is_file():
        rows = read_jsonl(output / "candidate_pool.jsonl")
        if (
            len(rows) != expected_rows
            or {str(row.get("parent_id")) for row in rows} != set(parent_ids)
            or any(
                row.get("stage") != normalized_stage
                or int(row.get("shard_index", -1)) != int(shard_index)
                for row in rows
            )
        ):
            raise ValueError("Pool shard resume candidate artifact is incomplete or foreign")
        for row in rows:
            assert_no_rf_provenance(row)
    else:
        generated = _generated_candidates(
            parents,
            base_model_path=base_model,
            adapter_path=adapter_path,
            config=config,
        )
        oracle = build_oracle(
            dataset=DATASET,
            backend=ORACLE_BACKEND,
            checkpoint=checkpoint,
            device=device,
            batch_size=oracle_batch_size,
        )
        rows = _score_generated_candidates(
            generated,
            oracle=oracle,
            featurizer=MolecularGraphFeaturizer(schema),
            stage=normalized_stage,
            shard_index=int(shard_index),
            oracle_batch_size=int(oracle_batch_size),
            checkpoint_id=str(card["checkpoint_id"]),
        )
        atomic_jsonl(output / "candidate_pool.jsonl", rows)
    expected_rows = len(parents) * config.num_return_sequences
    if len(rows) != expected_rows:
        raise RuntimeError(f"Pool shard row count {len(rows)} != expected {expected_rows}")
    manifest = {
        "schema_version": "bace_frozen_gnn_pool_shard_v1",
        "dataset": DATASET,
        "stage": normalized_stage,
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "split_used": "train",
        "train_only": True,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "shard_index": int(shard_index),
        "num_shards": NUM_SHARDS,
        "shard_rule": "sorted(parent_id)_position_mod_4",
        "all_parent_ids_sha256": stable_sha256(
            sorted(parent.parent_id for parent in all_parents)
        ),
        "parent_ids": parent_ids,
        "parent_ids_sha256": stable_sha256(parent_ids),
        "parent_shard_manifest_identity": parent_shard_identity,
        "parent_count": len(parents),
        "candidate_row_count": len(rows),
        "parse_ok_count": sum(bool(row["parse_ok"]) for row in rows),
        "direct_substructure_count": sum(bool(row["direct_substructure"]) for row in rows),
        "gnn_scored_deletion_count": sum(bool(row["oracle_ok"]) for row in rows),
        "strict_flip_count": sum(bool(row["cf_flip"]) for row in rows),
        "generation_config": asdict(config),
        "generation_config_hash": stable_sha256(asdict(config)),
        "policy_identity": policy_identity,
        "policy_checkpoint_hash": policy_identity["declared_policy_hash"],
        "oracle_checkpoint_hash": card["checkpoint_id"],
        "gnn_sha256sums_sha256": sha256_file(checkpoint / "sha256sums.txt"),
        "train_split_identity": file_identity(train_path),
        "candidate_pool_identity": file_identity(output / "candidate_pool.jsonl"),
        "created_at": utc_now(),
        "resume_supported": True,
        "execution_fingerprint": execution_fingerprint,
    }
    atomic_json(output / "pool_manifest.json", manifest)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: manifest[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "oracle_checkpoint_hash",
                "calibration_loaded",
                "calibration_dataset_loaded",
                "frozen_temperature_calibration_loaded",
                "test_loaded",
            )
        },
    )
    atomic_json(
        output / "IN_PROGRESS.json",
        {
            "schema_version": "bace_frozen_gnn_pool_resume_v1",
            "status": "COMPLETE",
            "stage": normalized_stage,
            "shard_index": int(shard_index),
            "execution_fingerprint": execution_fingerprint,
            "candidate_pool_identity": manifest["candidate_pool_identity"],
            "completed_at": utc_now(),
        },
    )
    atomic_marker(output / "PASS", "PASS")
    return manifest


def _canonical_candidate_id(fragment: str) -> str:
    return "BACE_RULE_" + stable_sha256({"fragment": fragment})[:24].upper()


def merge_pool_shards(
    *,
    shard_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    if len(shard_dirs) != 2 * NUM_SHARDS:
        raise ValueError("B10 requires exactly four B8 and four B9 shard roots")
    shard_entries: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    identities: dict[str, set[str]] = {
        "policy": set(),
        "oracle": set(),
        "parents": set(),
    }
    generation_configs: dict[str, set[str]] = {stage: set() for stage in POOL_STAGES}
    parents_by_stage: dict[str, set[str]] = {stage: set() for stage in POOL_STAGES}
    for path_like in shard_dirs:
        root = Path(path_like).expanduser().resolve(strict=True)
        manifest = validate_pass_manifest(
            root / "pool_manifest.json", require_no_test=True
        )
        stage = str(manifest.get("stage"))
        shard_index = int(manifest.get("shard_index", -1))
        key = (stage, shard_index)
        if stage not in POOL_STAGES or not 0 <= shard_index < NUM_SHARDS:
            raise ValueError(f"Invalid B8/B9 shard identity: {key}")
        if key in shard_entries:
            raise ValueError(f"Duplicate B8/B9 shard: {key}")
        if int(manifest.get("num_shards", 0)) != NUM_SHARDS:
            raise ValueError("Pool shard count is not frozen to four")
        assert_no_rf_provenance(manifest)
        shard_entries[key] = (root, manifest)
        identities["policy"].add(str(manifest.get("policy_checkpoint_hash")))
        identities["oracle"].add(str(manifest.get("oracle_checkpoint_hash")))
        identities["parents"].add(str(manifest.get("all_parent_ids_sha256")))
        generation_configs[stage].add(str(manifest.get("generation_config_hash")))
        parent_ids = {str(value) for value in manifest.get("parent_ids", [])}
        if parents_by_stage[stage] & parent_ids:
            raise ValueError(f"{stage} shards contain duplicate parents")
        parents_by_stage[stage].update(parent_ids)
        rows = read_jsonl(root / "candidate_pool.jsonl")
        declared_pool_identity = manifest.get("candidate_pool_identity")
        if not isinstance(declared_pool_identity, Mapping) or dict(
            declared_pool_identity
        ) != file_identity(root / "candidate_pool.jsonl"):
            raise ValueError(f"Pool shard candidate bytes differ from manifest: {root}")
        if len(rows) != int(manifest.get("candidate_row_count", -1)):
            raise ValueError(f"Pool shard row count differs from manifest: {root}")
        if any(
            row.get("stage") != stage
            or int(row.get("shard_index", -1)) != shard_index
            or str(row.get("parent_id")) not in parent_ids
            for row in rows
        ):
            raise ValueError(f"Pool shard rows escaped their fixed assignment: {root}")
        all_rows.extend(rows)
    expected_keys = {
        (stage, shard_index)
        for stage in POOL_STAGES
        for shard_index in range(NUM_SHARDS)
    }
    if set(shard_entries) != expected_keys:
        raise ValueError("B10 input shard set is incomplete")
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError(f"B8/B9 provenance identities differ: {identities}")
    if any(len(values) != 1 or "" in values for values in generation_configs.values()):
        raise ValueError(
            f"Pool generation configs differ within a frozen stage: {generation_configs}"
        )
    if parents_by_stage[POOL_STAGES[0]] != parents_by_stage[POOL_STAGES[1]]:
        raise ValueError("B8 and B9 did not cover the same frozen train parent set")

    output = fresh_output_dir(output_dir)
    best_by_parent_fragment: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_rows:
        fragment = canonicalize_smiles(str(row.get("final_fragment") or ""))
        parent_id = str(row.get("parent_id") or "")
        if not fragment or not parent_id:
            continue
        key = (parent_id, fragment)
        metric = row.get("reward_total")
        score = float(metric) if metric is not None and math.isfinite(float(metric)) else -math.inf
        old = best_by_parent_fragment.get(key)
        if old is None:
            best_by_parent_fragment[key] = {**row, "final_fragment": fragment}
            continue
        old_metric = old.get("reward_total")
        old_score = (
            float(old_metric)
            if old_metric is not None and math.isfinite(float(old_metric))
            else -math.inf
        )
        if (score, str(row.get("candidate_id"))) > (
            old_score,
            str(old.get("candidate_id")),
        ):
            best_by_parent_fragment[key] = {**row, "final_fragment": fragment}
    merged = sorted(
        best_by_parent_fragment.values(),
        key=lambda row: (
            str(row["parent_id"]),
            str(row["final_fragment"]),
            str(row["candidate_id"]),
        ),
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in merged:
        if not (
            row.get("parse_ok")
            and row.get("valid")
            and row.get("connected")
            and row.get("direct_substructure")
            and row.get("oracle_ok")
        ):
            continue
        grouped.setdefault(str(row["final_fragment"]), []).append(row)
    universe: list[dict[str, Any]] = []
    for fragment in sorted(grouped):
        source_rows = grouped[fragment]
        rewards = [float(row["reward_total"]) for row in source_rows]
        drops = [float(row["cf_drop"]) for row in source_rows]
        universe.append(
            {
                "candidate_id": _canonical_candidate_id(fragment),
                "canonical_fragment": fragment,
                "final_fragment": fragment,
                "source_parent_count": len({row["parent_id"] for row in source_rows}),
                "source_parent_ids": sorted({str(row["parent_id"]) for row in source_rows}),
                "source_reward_mean": float(np.mean(rewards)),
                "source_cf_drop_mean": float(np.mean(drops)),
                "source_strict_flip_count": sum(bool(row["cf_flip"]) for row in source_rows),
                "oracle_backend": ORACLE_BACKEND,
                "classifier_type": CLASSIFIER_TYPE,
                "rf_oracle_used": False,
            }
        )
    if len(universe) < 20:
        raise RuntimeError(
            f"B10 needs at least 20 structurally/GNN valid rules, found {len(universe)}"
        )
    atomic_jsonl(output / "candidate_pool.jsonl", merged)
    atomic_jsonl(output / "candidate_universe.jsonl", universe)
    policy_hash = next(iter(identities["policy"]))
    oracle_hash = next(iter(identities["oracle"]))
    manifest = {
        "schema_version": "bace_frozen_gnn_pool_merge_v1",
        "dataset": DATASET,
        "stage": "B10_POOL_MERGED",
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "split_used": "train",
        "train_only": True,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "policy_checkpoint_hash": policy_hash,
        "oracle_checkpoint_hash": oracle_hash,
        "input_shards": [
            {
                "stage": stage,
                "shard_index": index,
                "manifest": file_identity(root / "pool_manifest.json"),
                "candidate_pool": file_identity(root / "candidate_pool.jsonl"),
            }
            for (stage, index), (root, _manifest) in sorted(shard_entries.items())
        ],
        "input_row_count": len(all_rows),
        "merged_row_count": len(merged),
        "candidate_universe_count": len(universe),
        "parent_count": len(parents_by_stage[POOL_STAGES[0]]),
        "candidate_pool_hash": sha256_file(output / "candidate_pool.jsonl"),
        "candidate_universe_hash": sha256_file(output / "candidate_universe.jsonl"),
        "deterministic_merge": True,
        "created_at": utc_now(),
    }
    atomic_json(output / "merge_manifest.json", manifest)
    atomic_json(output / "pool_manifest.json", manifest)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: manifest[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "oracle_checkpoint_hash",
                "calibration_loaded",
                "calibration_dataset_loaded",
                "frozen_temperature_calibration_loaded",
                "test_loaded",
            )
        },
    )
    atomic_marker(output / "PASS", "PASS")
    return manifest


__all__ = [
    "BACEPoolConfig",
    "POOL_STAGES",
    "merge_pool_shards",
    "run_pool_shard",
]
