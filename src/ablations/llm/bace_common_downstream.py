"""Executable proposal-only BACE ablation, using the main scientific kernels.

This is not a matrix publisher.  Each variant has a separate output root and an
LLM-only result index.  Training attempts, including invalid strings and BRICS
shortfalls, remain in the denominator.  The frozen main BACE evaluator uses all
true-source parents, not the correctly-predicted/native GNN-ablation subset.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import math
from pathlib import Path
import signal
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from src.ablations.gnn import cpu_evaluation as evaluation
from src.ablations.gnn.cpu_training import load_bundle
from src.ablations.llm.bace_native_runtime import VARIANTS, verified_file
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.runtime_evidence import load_bace_reference_v2
from src.eval.bace_frozen_gnn_contracts import (
    atomic_csv, atomic_json, atomic_jsonl, load_bace_parents, read_json,
    read_jsonl, sha256_file, stable_sha256,
)

SCHEMA = "bace_llm_common_downstream_v1"
COHORT = "all_true_source_label_1_parents_as_main_BACE_load_bace_parents"


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"BACE_DOWNSTREAM_CONTRACT_MISMATCH:{label}")


def bind_downstream(bundle: Path, manifest: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    """Verify bytes only; do not parse calibration/test molecules here."""
    for split, expected in frozen["dataset_split_hashes"].items():
        _equal(sha256_file(evaluation.bundle_file(bundle, manifest, manifest["splits"][split])), expected, split)
    gine = bundle / manifest["gine_reference_root"]
    _equal(sha256_file(gine / "model.pt"), frozen["gine_checkpoint_sha"], "GINE")
    _equal(sha256_file(gine / "temperature_scaling.json"), frozen["temperature_sha"], "temperature")
    _equal(sha256_file(evaluation._input(bundle, manifest, "molclr_checkpoint_path")), frozen["molclr_sha"], "MolCLR")
    _equal(manifest["wnode_config"], frozen["wnode_config"], "WNode")
    aliases = {"selector_manifest": "frozen_selection_manifest_path",
               "thresholds": "thresholds_path", "variant_configs": "selector_variant_configs_path"}
    for role, key in aliases.items():
        _equal(sha256_file(evaluation._input(bundle, manifest, key)),
               frozen["selector_contract"][role]["sha256"], role)
    _equal(frozen["selector_contract"]["test_used_for_selection"], False, "test_selection")
    _equal((frozen["selector_contract"]["K"], frozen["selector_contract"]["Table2_K"]), (20, 10), "K")


def validate_attempts(rows: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]],
                      variant: str, *, brics: bool = False) -> None:
    expected = {}
    for call in calls:
        offset = 0 if call["regime"] == "base" else 4
        for index in range(4):
            key = (str(call["parent_id"]), offset + index)
            if key in expected:
                raise ValueError("DUPLICATE_PROPOSAL_CALL")
            expected[key] = call
    seen = set()
    for row in rows:
        key = (str(row["parent_id"]), int(row["attempt_index"]))
        if key not in expected or key in seen:
            raise ValueError("PROPOSAL_ATTEMPT_MISSING_DUPLICATED_OR_OUTSIDE_COHORT")
        seen.add(key)
        call = expected[key]
        _equal(row["variant"], variant, "attempt_variant")
        _equal(row["source_label"], 1, "source_label")
        if brics:
            if row.get("proposal_shortfall") and row.get("fragment_smiles"):
                raise ValueError("BRICS_SHORTFALL_BACKFILLED")
            for key2 in ("oracle_used", "calibration_loaded", "test_loaded"):
                _equal(row[key2], False, key2)
        else:
            for key2 in ("parent_smiles", "regime", "shard_id"):
                _equal(row[key2], call[key2], key2)
            _equal(row.get("train_only"), True, "train_only")
    if seen != set(expected):
        raise ValueError("PROPOSAL_ATTEMPT_COUNT_MISMATCH")


def load_attempts(spec: Mapping[str, Any], candidate_root: Path, reference_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant = spec["variant"]
    if variant == "BRICS_FIXED":
        adopted = spec["adopted_brics"]
        files = {name: verified_file(identity) for name, identity in adopted.items()}
        for path in files.values():
            path.resolve().relative_to(candidate_root)
        proposal = read_json(files["brics_proposal_manifest.json"])
        vocabulary = read_json(files["brics_vocab_manifest.json"])
        shortfall = read_json(files["brics_proposal_shortfall_receipt.json"])
        for key, filename in (("vocabulary_manifest", "brics_vocab_manifest.json"),
                              ("shortfall_receipt", "brics_proposal_shortfall_receipt.json")):
            _equal(verified_file(proposal[key]).resolve(), files[filename].resolve(), "BRICS_manifest_chain")
        for obj in (proposal, vocabulary, shortfall):
            _equal(obj["status"], "PASS", "BRICS_adoption")
            _equal(obj["calibration_loaded"], False, "BRICS_calibration")
            _equal(obj["test_loaded"], False, "BRICS_test")
        for obj in (vocabulary, shortfall):
            _equal(obj["reference_contract"]["sha256"], reference_sha, "BRICS_reference")
        _equal(vocabulary["source_split"], "train", "BRICS_vocabulary_split")
        _equal(vocabulary["oracle_fields_read"], [], "BRICS_oracle_ranking")
        _equal(proposal["oracle_used"], False, "BRICS_oracle")
        _equal(shortfall["candidate_duplication_used"], False, "BRICS_duplication")
        _equal(shortfall["oracle_ranking_used"], False, "BRICS_oracle_ranking")
        _equal(shortfall["shortfall_is_not_backfilled"], True, "BRICS_shortfall")
        pool = verified_file(proposal["candidate_pool"])
        _equal(pool.resolve(), files["brics_proposal_pool.jsonl"].resolve(), "BRICS_pool")
        attempts_file = verified_file(proposal["attempt_records"])
        attempts_file.resolve().relative_to(candidate_root)
        _equal(shortfall["attempt_records"], proposal["attempt_records"], "BRICS_attempts")
        attempts = read_jsonl(attempts_file)
        emitted = read_jsonl(pool)
        projected = {(str(row["parent_id"]), int(row["attempt_index"])): row["fragment_smiles"]
                     for row in attempts if not row["proposal_shortfall"]}
        observed = {(str(row["parent_id"]), int(row["attempt_index"])): row["fragment_smiles"] for row in emitted}
        if projected != observed or len(observed) != len(emitted):
            raise ValueError("BRICS_POOL_ATTEMPT_BINDING_MISMATCH")
        evidence = {"kind": "BRICS_EXISTING_TRAIN_ONLY_POOL", "attempts_sha256": sha256_file(attempts_file),
                    "pool_sha256": sha256_file(pool)}
    else:
        receipt = read_json(candidate_root / "candidate_generation_receipt.json")
        _equal(receipt["status"], "CANDIDATE_POOL_PASS", "generator_terminal")
        _equal(receipt["spec_sha256"], canonical_json_sha256(spec), "generator_spec")
        _equal(receipt["variant"], variant, "generator_variant")
        _equal(receipt["next_call"], len(spec["calls"]), "generator_cursor")
        for key in ("test_loaded", "calibration_loaded", "training_performed"):
            _equal(receipt[key], False, key)
        pool = candidate_root / "candidate_pool.jsonl"
        _equal(sha256_file(pool), receipt["candidate_pool_sha256"], "generated_pool")
        attempts = read_jsonl(pool)
        _equal(receipt["proposal_attempts"], len(attempts), "proposal_attempts")
        evidence = {"kind": "MATCHED_NATIVE_REGEN", "pool_sha256": sha256_file(pool),
                    "receipt_sha256": sha256_file(candidate_root / "candidate_generation_receipt.json")}
    validate_attempts(attempts, spec["calls"], variant, brics=variant == "BRICS_FIXED")
    return attempts, evidence


def merge_scored_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The main B10 deterministic merge, without the eight-shard I/O wrapper.

    Deliberately does not prefilter on train strict flips.  Source reward is
    only the original per-parent duplicate tie-break and not calibration order.
    """
    from src.eval.molclr_node_embeddings import canonicalize_smiles
    from src.eval.bace_frozen_gnn_pool import _canonical_candidate_id
    best = {}
    for row in rows:
        fragment = canonicalize_smiles(str(row.get("final_fragment") or ""))
        if not fragment or not row.get("parent_id"):
            continue
        key = (str(row["parent_id"]), fragment)
        def rank(value):
            reward = value.get("reward_total")
            return (float(reward) if reward is not None and math.isfinite(float(reward)) else -math.inf,
                    str(value.get("candidate_id")))
        if key not in best or rank(row) > rank(best[key]):
            best[key] = {**row, "final_fragment": fragment}
    merged = sorted(best.values(), key=lambda r: (r["parent_id"], r["final_fragment"], r["candidate_id"]))
    grouped = {}
    for row in merged:
        if all(row.get(k) for k in ("parse_ok", "valid", "connected", "direct_substructure", "oracle_ok")):
            grouped.setdefault(row["final_fragment"], []).append(row)
    universe = [{"candidate_id": _canonical_candidate_id(fragment), "canonical_fragment": fragment,
        "final_fragment": fragment, "source_parent_count": len({r["parent_id"] for r in source}),
        "source_parent_ids": sorted({str(r["parent_id"]) for r in source}),
        "source_reward_mean": float(np.mean([r["reward_total"] for r in source])),
        "source_cf_drop_mean": float(np.mean([r["cf_drop"] for r in source])),
        "source_strict_flip_count": sum(bool(r["cf_flip"]) for r in source),
        "oracle_backend": "gnn", "classifier_type": "gnn", "rf_oracle_used": False}
        for fragment, source in sorted(grouped.items())]
    return merged, universe


def candidate_metrics(scored: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(attempts)
    if len(scored) != denominator or not denominator:
        raise ValueError("All attempts, including failures and shortfalls, must be scored")
    values = {f"{name}_rate": sum(bool(row.get(key)) for row in scored) / denominator
              for name, key in (("parse", "parse_ok"), ("valid", "valid"), ("connected", "connected"),
                               ("direct_substructure", "direct_substructure"), ("strict_flip", "cf_flip"))}
    drops = [float(row["cf_drop"]) for row in scored if row.get("cf_drop") is not None and math.isfinite(float(row["cf_drop"]))]
    return {**values, "proposal_attempts": denominator, "projection_rate": 0.0,
        "projection_enabled": False, "projection_policy": "MAIN_B8_B9_PROJECTION_DISABLED",
        "proposal_shortfall": sum(bool(row.get("proposal_shortfall")) for row in attempts),
        "mean_cf_drop_valid": float(np.mean(drops)) if drops else None,
        "parents_with_valid": len({r["parent_id"] for r in scored if r.get("valid")}),
        "parents_with_flip": len({r["parent_id"] for r in scored if r.get("cf_flip")}),
        "unique_fragment_count": len({r["final_fragment"] for r in scored if r.get("valid")})}


def _index_result(registry_root: Path, variant: str, output: Path, audit: Mapping[str, Any]) -> None:
    if "fast16_matrix_authority" in registry_root.parts or "control" in registry_root.parts:
        raise ValueError("LLM_REGISTRY_MUST_NOT_BE_MAIN_CONTROL_AUTHORITY")
    registry_root.mkdir(parents=True, exist_ok=True)
    with (registry_root / "llm_registry.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = registry_root / "llm_result_registry.json"
        registry = read_json(path) if path.exists() else {"schema_version": "bace_llm_independent_results_v1", "results": {}}
        _equal(registry["schema_version"], "bace_llm_independent_results_v1", "registry_schema")
        identity = {"output_root": str(output), "audit_sha256": sha256_file(output / "final_audit.json"),
                    "binding_sha256": audit["binding_sha256"], "state": "PASS"}
        if variant in registry["results"] and registry["results"][variant] != identity:
            raise ValueError("LLM_REGISTRY_VARIANT_CONFLICT")
        registry["results"][variant] = identity
        atomic_json(path, registry)


def _sealed_json(path: Path, payload: Mapping[str, Any]) -> None:
    sealed = {**payload, "self_sha256": stable_sha256(payload)}
    if path.exists():
        _equal(read_json(path), sealed, "resume_sealed_artifact")
    else:
        atomic_json(path, sealed)


def _heldout(*, bundle: Path, manifest: Mapping[str, Any], output: Path, universe: list[dict[str, Any]],
             selector: Mapping[str, Any], oracle: Any, featurizer: Any, distance: Any,
             binding: str, batch_size: int, pause: Any) -> dict[str, Any]:
    matrices, selected = {}, None
    for split in ("calibration", "test"):
        if pause():
            return {"state": "PAUSED_AT_SAFE_PARENT_BOUNDARY", "split": split}
        if split == "test":
            freeze = read_json(output / "selector_manifest.json")
            _equal(freeze["self_sha256"], stable_sha256({k: v for k, v in freeze.items() if k != "self_sha256"}), "freeze_hash")
            _equal(freeze["binding_sha256"], binding, "freeze_binding")
            _equal(freeze["test_loaded"], False, "pre_test_freeze")
            by_id = {row["candidate_id"]: row for row in universe}
            selected = [by_id[key] for key in freeze["ordered_rule_ids"]]
        candidates = universe if split == "calibration" else selected
        # Main-route scientific cohort: no correctly-predicted prefilter.
        try:
            parents = load_bace_parents(evaluation.bundle_file(bundle, manifest, manifest["splits"][split]), source_label=1)
        except ValueError as exc:
            if not str(exc).startswith("No BACE source-label=1 parents found:"):
                raise
            parents = []
        if split == "calibration" and not parents:
            raise ValueError("BLOCKED_EMPTY_CALIBRATION_COHORT")
        predictions = evaluation._predict(parents, oracle, featurizer, split, batch_size)
        by_parent = {p.parent_id: row for p, row in zip(parents, predictions, strict=True)}
        atomic_json(output / f"{split}_cohort_manifest.json", {"definition": COHORT,
            "source_label": 1, "parent_ids": [p.parent_id for p in parents],
            "split_sha256": manifest["files"][manifest["splits"][split]]["sha256"]})
        pairs = []
        for parent in parents:
            if pause():
                return {"state": "PAUSED_AT_SAFE_PARENT_BOUNDARY", "split": split}
            pairs.extend(evaluation._pairs([parent], candidates, oracle=oracle, featurizer=featurizer,
                distance=distance, split=split, output=output / "parent_checkpoints" / split,
                binding=binding, batch_size=batch_size, predictions=by_parent))
            atomic_json(output / "progress.json", {"state": "RUNNING", "phase": split,
                "completed_pairs": len(pairs), "expected_pairs": len(parents) * len(candidates)})
        atomic_jsonl(output / f"{split}_pairs.jsonl", pairs)
        matrix = evaluation.matrix_from_pairs([p.parent_id for p in parents], candidates, pairs, root=output, split=split)
        matrices[split] = matrix
        if split == "calibration":
            sequence, trace = evaluation.select_calibration(matrix, selector)
            if len(sequence) != 20:
                raise ValueError("SCIENTIFIC_FAILED_INSUFFICIENT_VALID_UNIQUE_RULES")
            frozen = {"schema_version": "bace_llm_calibration_selector_v1", "binding_sha256": binding,
                "selection_frozen": True, "test_loaded": False, "cohort_definition": COHORT,
                "selector_input_sha256": selector["input_sha256"], "calibration_pairs_sha256": sha256_file(output / "calibration_pairs.jsonl"),
                "ordered_rule_ids": [universe[i]["candidate_id"] for i in sequence], "trace": trace}
            _sealed_json(output / "selector_manifest.json", frozen)
    metrics = evaluation.explanation_metrics(matrices["test"], range(20), selector["thresholds"])
    matrix = matrices["test"]
    theta = selector["thresholds"].theta_star
    if matrix.parent_ids:
        rng = np.random.default_rng(7)
        statistics = {"CCRCov@10": [], "CCRCov@20": []}
        for _ in range(1000):
            indices = rng.integers(0, len(matrix.parent_ids), size=len(matrix.parent_ids))
            for k in (10, 20):
                statistics[f"CCRCov@{k}"].append(float(np.mean(np.min(matrix.distances[indices, :k], axis=1) <= theta)))
        ci = {key: {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975))}
              for key, values in statistics.items()}
    else:
        ci = {"CCRCov@10": None, "CCRCov@20": None}
    atomic_json(output / "bootstrap_parent_ci.json", {"resamples": 1000, "seed": 7,
        "scope": "within_seed_test_parent_resampling_only", "selector_refit": False,
        "across_seed_standard_deviation_claimed": False, "confidence": .95, "intervals": ci})
    atomic_json(output / "heldout_test_metrics.json", metrics)
    atomic_json(output / "selected_rules.json", {"rules": selected, "selector_sha256": sha256_file(output / "selector_manifest.json")})
    prefix_rows = metrics["prefix_rows"] or [{"k": k, "ccrcov_theta_star": None,
        "state": "VALID_EMPTY_COHORT", "conditional_median_cost": None} for k in range(1, 21)]
    threshold_rows = metrics["threshold_rows"] or [{"threshold": float(x), "CCRCov": None,
        "state": "VALID_EMPTY_COHORT", "K": 20} for x in sorted(set(selector["thresholds"].raw_thresholds))]
    atomic_csv(output / "figure3_coverage_vs_k.csv", prefix_rows)
    atomic_csv(output / "figure4_coverage_vs_threshold.csv", threshold_rows)
    atomic_csv(output / "table2_k10.csv", [row for row in prefix_rows if int(row["k"]) == 10])
    return {"state": "PASS", "metrics": metrics}


def run_downstream(*, task_spec: str | Path, candidate_root: str | Path, gnn_input_bundle: str | Path,
                   gnn_verified_archive: str | Path, gnn_verified_sha256: str, registry_root: str | Path,
                   output_root: str | Path, resume: bool = False, device: str = "cpu", batch_size: int = 64,
                   cpu_threads: int = 2) -> dict[str, Any]:
    from src.ablations.gnn.scientific_verification import verify_package_archive
    from src.ablations.llm.bace_readiness import generation_calls
    from src.eval.bace_frozen_gnn_pool import _score_generated_candidates
    from src.oracles.gnn_oracle import GNNOracle
    import torch

    # No model is loaded and no science/output starts before independent GNN PASS.
    archive = Path(gnn_verified_archive).resolve(strict=True)
    _equal(sha256_file(archive), gnn_verified_sha256, "independent_GNN_archive")
    gnn_pass = verify_package_archive(archive)
    _equal(gnn_pass["state"], "PASS", "independent_GNN_core")
    spec = read_json(Path(task_spec))
    _equal(spec["schema_version"], "bace_native_llm_task_v1", "spec_schema")
    if spec["variant"] not in VARIANTS:
        raise ValueError("UNKNOWN_LLM_VARIANT")
    _equal(spec["task_spec_sha256"], canonical_json_sha256({k: v for k, v in spec.items() if k != "task_spec_sha256"}), "task_self_hash")
    reference = load_bace_reference_v2(spec["reference_contract"]["path"], spec["reference_contract"]["sha256"])
    _equal(spec["downstream_contract"], reference.payload["frozen_downstream"], "native_downstream")
    _equal(spec["calls"], generation_calls(reference.payload), "native_call_contract")
    bundle, manifest = load_bundle(gnn_input_bundle)
    bind_downstream(bundle, manifest, reference.payload["frozen_downstream"])
    candidates_source = Path(candidate_root).resolve(strict=True)
    attempts, pool_evidence = load_attempts(spec, candidates_source, reference.file_sha256)
    output = Path(output_root).resolve()
    registry = Path(registry_root).resolve()
    for source in (bundle, candidates_source, archive.parent, Path(reference.path).parent):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("OUTPUT_MUST_BE_DISJOINT_FROM_IMMUTABLE_INPUTS")
    if registry == output or registry in output.parents or output in registry.parents:
        raise ValueError("REGISTRY_MUST_BE_SEPARATE_FROM_VARIANT_OUTPUT")
    if output.exists() and not resume:
        raise FileExistsError("Fresh output root or explicit same-root resume required")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cpu_threads)
    stopped = [False]
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    pause = lambda: stopped[0] or (output / "pause.request").is_file()
    try:
        with (output / "writer.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selector = evaluation.frozen_selector(bundle, manifest)
            sources = [Path(__file__), Path(evaluation.__file__), Path(__import__("src.eval.bace_frozen_gnn_pool", fromlist=["_"]).__file__),
                       Path(__import__("src.eval.bace_frozen_gnn_verification", fromlist=["_"]).__file__)]
            driver_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3], text=True).strip()
            contract = {"schema_version": SCHEMA, "variant": spec["variant"], "dataset": "bace", "method": "ours",
                "task_spec_sha256": sha256_file(Path(task_spec)), "reference_sha256": reference.file_sha256,
                "bundle_sha256": sha256_file(bundle / "bundle_manifest.json"), "pool": pool_evidence,
                "gnn_independent_core": gnn_pass, "selector_input_sha256": selector["input_sha256"],
                "scientific_sources": {p.name: sha256_file(p) for p in sources}, "device": device,
                "oracle_batch_size": batch_size, "cpu_threads": cpu_threads,
                "execution_commit": driver_commit, "generation_execution_commit": spec["execution_commit"],
                "gnn_input_scientific_commit": manifest["execution_commit"],
                "main_matrix_write": False, "training_performed": False,
                "cohort_definition": COHORT, "bootstrap_scope": "within_seed7_parent_only_not_across_seed_std"}
            binding = stable_sha256(contract)
            _sealed_json(output / "run_manifest.json", {**contract, "binding_sha256": binding})
            if (output / "final_audit.json").is_file():
                audit = read_json(output / "final_audit.json")
                for rel, digest in audit["files"].items():
                    _equal(sha256_file(output / rel), digest, "final_inventory")
                _index_result(registry, spec["variant"], output, audit)
                return audit
            oracle = GNNOracle.from_checkpoint(bundle / manifest["gine_reference_root"], device=device, batch_size=batch_size)
            _equal((oracle.backbone, oracle.source_label, oracle.num_classes), ("gine", 1, 2), "frozen_oracle")
            _equal(oracle.temperature, spec["downstream_contract"]["temperature"], "frozen_temperature")
            featurizer = evaluation._featurizer(bundle, manifest)
            parents = {p.parent_id: p for p in load_bace_parents(evaluation.bundle_file(bundle, manifest, manifest["splits"]["train"]), source_label=1)}
            by_parent = {}
            for row in attempts:
                by_parent.setdefault(str(row["parent_id"]), []).append(row)
            scored = []
            for parent_id, rows in sorted(by_parent.items()):
                if pause():
                    return {"state": "PAUSED_AT_SAFE_PARENT_BOUNDARY", "phase": "train_verification"}
                parent = parents[parent_id]
                key = stable_sha256({"binding": binding, "parent": asdict(parent), "attempts": rows})
                checkpoint = output / "parent_checkpoints" / "train" / f"{key}.json"
                if checkpoint.exists():
                    state = read_json(checkpoint)
                    _equal(state["self_sha256"], stable_sha256({k: v for k, v in state.items() if k != "self_sha256"}), "train_checkpoint")
                    _equal(state["binding_sha256"], binding, "train_checkpoint_binding")
                    current = state["rows"]
                else:
                    from src.models.llm_generator import clean_generated_smiles
                    if spec["variant"] != "BRICS_FIXED":
                        for row in rows:
                            _equal(row["fragment_smiles"], clean_generated_smiles(row["raw_text"]), "main_parser")
                    generated = [(parent, int(row["attempt_index"]), str(row.get("raw_text") or row.get("fragment_smiles") or ""),
                                  str(row.get("fragment_smiles") or "")) for row in rows]
                    current = _score_generated_candidates(generated, oracle=oracle, featurizer=featurizer,
                        stage="LLM_COMMON_TRAIN_ONLY", shard_index=0, oracle_batch_size=batch_size, checkpoint_id=oracle.checkpoint_id)
                    _sealed_json(checkpoint, {"binding_sha256": binding, "rows": current})
                scored.extend(current)
            merged, universe = merge_scored_rows(scored)
            atomic_jsonl(output / "scored_attempts.jsonl", scored)
            atomic_jsonl(output / "candidate_pool.jsonl", merged)
            atomic_jsonl(output / "candidate_universe.jsonl", universe)
            atomic_json(output / "candidate_metrics.json", candidate_metrics(scored, attempts))
            if len(universe) < 20:
                blocked = {"state": "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_UNIQUE_RULES", "valid_unique_rules": len(universe),
                           "required_rules": 20, "test_loaded": False, "candidate_padding_used": False}
                atomic_json(output / "terminal.json", blocked)
                return blocked
            atomic_json(output / "verification_manifest.json", {"state": "TRAIN_CANDIDATES_FROZEN", "test_loaded": False,
                "calibration_loaded": False, "candidate_universe_sha256": sha256_file(output / "candidate_universe.jsonl"),
                "binding_sha256": binding, "attempt_count": len(attempts), "universe_count": len(universe)})
            distance = evaluation._distance(bundle, manifest, output)
            try:
                result = _heldout(bundle=bundle, manifest=manifest, output=output, universe=universe,
                    selector=selector, oracle=oracle, featurizer=featurizer, distance=distance,
                    binding=binding, batch_size=batch_size, pause=pause)
            finally:
                distance.close()
            if result["state"] != "PASS":
                return result
            files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*"))
                     if p.is_file() and "cache" not in p.relative_to(output).parts and p.name not in {"writer.lock", "pause.request", "progress.json", "terminal.json"}}
            audit = {"schema_version": SCHEMA, "state": "PASS", "binding_sha256": binding, "files": files,
                "calibration_selection_before_test": True, "test_selection": False, "main_matrix_write": False,
                "valid_zero_result": result["metrics"].get("strict_flip_rate") == 0,
                "evaluation_scope": "one_frozen_GINE_proposal_generator_only", "gnn_core_verified": True}
            atomic_json(output / "final_audit.json", audit)
            _index_result(registry, spec["variant"], output, audit)
            atomic_json(output / "terminal.json", audit)
            return audit
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
