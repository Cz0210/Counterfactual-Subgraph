"""Read-only BACE provenance adoption and concrete native-generator task specs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.ablations.llm.bace_native_runtime import (
    SOURCE_PINS, VARIANTS, audit_native_source, verified_file, verified_2b_runtime_proof,
)
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.runtime_evidence import load_bace_reference_v2
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, file_identity, fixed_parent_shard_map, load_bace_parents, sha256_file,
)


def generation_calls(reference: Mapping[str, Any], *, file_resolver=verified_file) -> list[dict[str, Any]]:
    generation = reference["candidate_generation"]
    train = file_resolver({"path": reference["frozen_downstream"]["dataset_split_paths"]["train"],
                          "sha256": reference["frozen_downstream"]["dataset_split_hashes"]["train"]})
    cohort = json.loads(file_resolver(generation["parent_manifest"]).read_text())
    ids = cohort.get("parent_ids", [])
    if (cohort.get("schema_version") != "bace_frozen_parent_ids_v1"
            or cohort.get("split") != "train" or cohort.get("source_label") != 1
            or len(ids) != 386 or len(set(ids)) != 386 or ids != sorted(ids)
            or generation.get("attempts_per_parent") != 8
            or cohort.get("split_identity", {}).get("sha256") != sha256_file(train)):
        raise ValueError("BACE proposal cohort/attempt contract differs")
    parents = {p.parent_id: p for p in load_bace_parents(train, source_label=1)}
    if any(parent_id not in parents for parent_id in ids):
        raise ValueError("Frozen proposal cohort is not a train source-class subset")
    shard_map = fixed_parent_shard_map(ids)
    calls = []
    for shard in range(4):
        for regime, field in (("base", "base_regime"), ("high_temperature", "high_temperature_regime")):
            values = generation[field]
            if (values["num_return_sequences"], values["max_new_tokens"], values["batch_size"], values["top_p"]) != (4, 96, 1, 0.9):
                raise ValueError("Frozen four-sequence decoding contract differs")
            for parent_id in ids:
                if shard_map[parent_id] != shard:
                    continue
                parent = parents[parent_id]
                # Preserve the exact audited task text; only native role wrappers vary.
                prompt = reference["prompt"]["template_prefix"] + "\n" + "\n".join(
                    [f"ORIGINAL_LABEL: {parent.label}", f"MOLECULE_SMILES: {parent.smiles}", "FRAGMENT_SMILES:"])
                calls.append({"parent_id": parent_id, "parent_smiles": parent.smiles, "shard_id": shard,
                    "regime": regime, "seed": values["seed"], "temperature": values["temperature"],
                    "prompt": prompt})
    return calls


def ppo_adoption_decision(reference: Mapping[str, Any], *, rendering: str) -> dict[str, Any]:
    """Do not label the old plain-prompt candidate pool a native-chat match."""
    differences = []
    if reference["prompt"]["rendering"] != rendering:
        differences.append("PROMPT_RENDERING_CHANGED_FROM_MAIN_PLAIN_TO_NATIVE_CHAT")
    return {"state": "MATCHED_REGEN_REQUIRED" if differences else "REQUIRES_COMPLETE_POOL_DOWNSTREAM_BINDING",
            "matching_failures": differences, "checkpoint_reused": True,
            "checkpoint": reference["ppo"]["checkpoint_root"], "training_required": False,
            "optimizer_updates_already_performed": reference["ppo"]["optimizer_updates"],
            "project_sft_checkpoint_exists": False, "main_science_automatically_adopted": False}


def _small_source_inventory(root: Path, expected_weights: Mapping[str, str]) -> dict[str, str]:
    files = {path.name: sha256_file(path) for path in root.iterdir()
             if path.is_file() and not path.name.endswith((".safetensors", ".bin"))}
    files.update(expected_weights)
    return files


def prepare_bace_llm(*, reference_path: str | Path, reference_sha256: str,
                     two_b_root: str | Path, brics_root: str | Path,
                     output_root: str | Path, execution_commit: str,
                     two_b_isolated_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reference = load_bace_reference_v2(reference_path, reference_sha256)
    values = reference.payload
    calls = generation_calls(values)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    base_root = Path(values["base_model"]["path"])
    model_specs, blockers = {}, {}
    for size, root in (("7b", base_root), ("2b", Path(two_b_root))):
        try:
            audit = audit_native_source(root, size)
            if size == "7b":
                weights = {name: row["local_sha256"] for name, row in values["base_model"]["weight_binding"].items()}
            else:
                snapshot = json.loads((root / "snapshot_manifest.json").read_text())
                if snapshot.get("status") != "PASS" or snapshot.get("revision") != SOURCE_PINS[size]["revision"]:
                    raise ValueError("2B snapshot is not the frozen downloaded revision")
                weights = {row["path"]: row["sha256"] for row in snapshot["inventory"]
                           if row["path"].endswith(".safetensors")}
            if not weights or any(not (root / name).is_file() for name in weights):
                raise ValueError("Cached model weights are missing; no download authorized here")
            model_specs[size] = {"root": str(root), "size": size, "revision": SOURCE_PINS[size]["revision"],
                "files": _small_source_inventory(root, weights), "remote_code_audit": audit,
                "native_prompt_api": "model.build_inputs(empty_history,empty_system)",
                "weight_hashes_reverified_on_actual_load": True, "quantization": "NF4_DOUBLE_QUANT_BF16_MAIN_MATCHED"}
            atomic_json(output / f"{size}_remote_code_audit.json", audit)
        except Exception as exc:
            blockers[size] = str(exc)
    ppo_files = {name: values["ppo"][key] for name, key in
                 (("adapter_config.json", "adapter_config"), ("adapter_model.safetensors", "adapter_weights"))}
    for identity in ppo_files.values():
        verified_file(identity)
    ppo = {"root": values["ppo"]["checkpoint_root"], "files": ppo_files,
           "optimizer_updates": 300, "training_rerun_allowed": False}
    common = {"schema_version": "bace_native_llm_task_v1", "execution_commit": execution_commit,
        "dataset": "bace", "method": "ours", "seed": 7, "source_label": 1,
        "output_scope_root": str(Path(reference.path).parent.parent / "llm"),
        "reference_contract": {"path": reference.path, "sha256": reference.file_sha256},
        "calls": calls, "parent_count": 386, "attempts_per_parent": 8,
        "main_adaptation": "BASE_PLUS_FRESH_LORA_PLUS_PPO", "project_sft_exists": False,
        "prompt_rendering": "PINNED_NATIVE_MODEL_BUILD_INPUTS",
        "rng_initialization": "ONCE_PER_SHARD_REGIME_AFTER_MODEL_LOAD",
        "historical_main_candidate_parity_claimed": False,
        "prompt_semantics_sha256": canonical_json_sha256(values["prompt"]),
        "downstream_contract": values["frozen_downstream"],
        "generation_train_split": {"path": values["frozen_downstream"]["dataset_split_paths"]["train"],
            "sha256": values["frozen_downstream"]["dataset_split_hashes"]["train"]},
        "created_during_cpu_preparation": True, "gnn_core_required_before_science": True,
        "main_matrix_13_required": False, "test_loaded_during_generation": False,
        "formal_safe_gpu_release_seconds_measured": None}
    tasks = {}
    project = Path(__file__).resolve().parents[3]
    downstream_entrypoint = project / "scripts/ablations/llm/run_bace_common_downstream.py"
    downstream_module = project / "src/ablations/llm/bace_common_downstream.py"
    for variant in VARIANTS:
        spec = {**common, "variant": variant, "ppo_adapter": ppo if variant == VARIANTS[2] else None}
        if variant == VARIANTS[0]:
            brics = Path(brics_root)
            names = ("brics_proposal_pool.jsonl", "brics_proposal_manifest.json", "brics_vocab_manifest.json",
                     "brics_proposal_shortfall_receipt.json")
            spec["model"] = None
            try:
                artifacts = {name: file_identity(brics / name) for name in names}
                from src.ablations.llm.core_execution import validate_variant_artifact_bindings, CoreLLMVariant
                from types import SimpleNamespace
                adopted = [SimpleNamespace(path=row["path"], sha256=row["sha256"]) for row in artifacts.values()]
                validate_variant_artifact_bindings(SimpleNamespace(variant=CoreLLMVariant.BRICS_FIXED,
                    stages=[SimpleNamespace(adopted_artifacts=adopted)]), reference)
                spec["adopted_brics"] = artifacts
                spec["generator_state"] = "TRAIN_ONLY_POOL_ADOPTION_READY"
            except Exception as exc:
                spec["generator_state"] = "BLOCKED_BRICS_ADOPTION"
                spec["blocker"] = str(exc)
        else:
            size = "2b" if variant == VARIANTS[3] else "7b"
            spec["model"] = model_specs.get(size)
            spec["generator_state"] = "LOADER_AND_RESUME_READY_WAITING_GNN_CORE" if size in model_specs else "BLOCKED_MODEL_SOURCE"
            spec["blocker"] = blockers.get(size)
            if size == "2b":
                spec["isolated_cpu_load_receipt_required_before_generation"] = True
                if spec["model"] is not None:
                    try:
                        verified_2b_runtime_proof(spec["model"], two_b_isolated_receipt)
                        spec["isolated_cpu_load_receipt"] = dict(two_b_isolated_receipt)
                    except Exception as exc:
                        spec["generator_state"] = "BLOCKED_MISSING_ISOLATED_CPU_PROOF"
                        spec["blocker"] = str(exc)
            if variant == VARIANTS[2]:
                spec["main_adoption_decision"] = ppo_adoption_decision(values, rendering=common["prompt_rendering"])
        # This is code-entrypoint readiness, never a result/metric PASS. The
        # real single-GINE adapter has focused tests; it still requires the
        # independently verified GNN package and real candidate pool at runtime.
        spec["downstream_state"] = "EXECUTABLE_ENTRYPOINT_READY_WAITING_GNN_CORE"
        spec["downstream_implementation"] = {"execution_commit": execution_commit,
            "entrypoint": file_identity(downstream_entrypoint),
            "module": file_identity(downstream_module),
            "cohort": "FROZEN_MAIN_GINE_TRUE_SOURCE_CORRECT_PREDICTION",
            "calibration_only_selector": True, "test_only_after_freeze": True}
        spec["task_spec_sha256"] = canonical_json_sha256(spec)
        path = output / f"{variant}.task.json"
        atomic_json(path, spec)
        tasks[variant] = {"path": str(path), "sha256": sha256_file(path),
                          "generator_state": spec["generator_state"], "downstream_state": spec["downstream_state"]}
    report = {"schema_version": "bace_llm_native_readiness_v1", "status": "PREPARATION_COMPLETE_WITH_BLOCKERS",
        "variants": tasks, "model_source_blockers": blockers, "reference_contract_sha256": reference.file_sha256,
        "ppo_adoption": ppo_adoption_decision(values, rendering=common["prompt_rendering"]),
        "science_started": False, "gpu_lock_acquired": False, "main_matrix_write": False}
    report["corrected_core_successors"] = {
        "gate": "INDEPENDENT_GNN_CORE_SEED7_CORRECTED_PASS",
        "cpu_l0_entrypoint": file_identity(project / "scripts/hpc/llm/run_bace_l0_cpu.py"),
        "gpu_entrypoint": file_identity(project / "scripts/ablations/llm/run_bace_llm_successor.py"),
        "gpu_generation_order": list(VARIANTS[1:]), "gpu_borrow_enabled": False,
        "max_early_llm_gpus": 1, "idle_gpu_seconds": 1200,
        "main_matrix_count_required": False, "secondary_seeds_required": False,
        "existing_owner_lease_required": True, "main_reservation_precedence": True,
    }
    atomic_json(output / "llm_readiness.json", report)
    return report
