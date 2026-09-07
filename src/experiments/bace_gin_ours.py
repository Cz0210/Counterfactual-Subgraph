"""Original BACE Ours66 operations under the accepted, frozen GIN oracle.

This is a method adapter, not a campaign, trainer, Reach selector or publisher.
The fixed ground-truth-source cohort is never reduced by GIN predictions.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any, Mapping

from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, load_bace_parents, read_json, sha256_file, stable_sha256,
)

SCOPE = "BACE_FIXED_POOL_FROZEN_GIN_V1"
ORIGINAL_UNIVERSE_SHA = "77fdb9f2243dc05271c7da653c609f39167eaf75aa7cc8c015aaf3d8af8b64ab"
ORIGINAL_BUNDLE_SHA = "4675937e5b86c1405a6fd14df71d3820847218d311821d86aec939baa92197d8"


def with_native_graph_distance(delegate, *, index, current_raw_contract, repo):
    """Expose exact graph-pair costs for native methods without fake delete keys."""
    from src.ablations.gnn.reach_raw_distance_reuse import VerifiedRawGraphDistance, graph_key

    class NativeGraphDistance(VerifiedRawGraphDistance):
        def distance(self, parent, counterfactual):
            key, p, cf = graph_key(parent, counterfactual, self.index["raw_contract_sha256"])
            record = self.index["graph_costs"].get(key)
            if record is not None:
                self.used.append(dict(raw_graph_key=key, source_index_sha256=self.index["self_sha256"],
                    current_native_graph_pair={"parent": p, "counterfactual": cf},
                    source_records=record["source_records"]))
                return dict(ok=True, distance=record["distance"], cache_hit=True, error=None,
                    metadata={"reuse": "EXPLICIT_GRAPH_CONTENT_RAW_OT_ADOPTION"})
            if key not in self.local:
                self.local[key] = self.delegate.distance(parent, counterfactual)
                self.fresh += not self.local[key].get("cache_hit", False)
            return self.local[key]

    return NativeGraphDistance(delegate, index=index, current_raw_contract=current_raw_contract, repo=repo)


def _bound_json(path: Path, digest: str) -> dict[str, Any]:
    if not digest or sha256_file(path) != digest:
        raise ValueError(f"BOUND_INPUT_CHANGED:{path}")
    return read_json(path)


def original_bundle(spec: Mapping[str, Any]):
    """Read the manifest and only subsequently consumed inputs, not all weights."""
    from src.ablations.contracts import canonical_json_sha256
    root = Path(spec["bundle_root"]).resolve(strict=True)
    expected = spec.get("bundle_manifest_sha256", ORIGINAL_BUNDLE_SHA)
    if expected != ORIGINAL_BUNDLE_SHA:
        raise ValueError("ORIGINAL_ACCEPTED_INPUT_BUNDLE_REQUIRED")
    manifest = _bound_json(root / "bundle_manifest.json", expected)
    if manifest.get("manifest_sha256") != canonical_json_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("ORIGINAL_BUNDLE_SELF_HASH_CHANGED")
    if (manifest.get("schema_version") != "bace_gnn_cpu_bundle_v1"
            or manifest.get("dataset") != "bace" or manifest.get("seed") != 7
            or manifest.get("num_classes") != 2):
        raise ValueError("ORIGINAL_BACE_SEED7_BUNDLE_REQUIRED")
    return root, manifest


def load_original_pool(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    from src.ablations.gnn.cpu_evaluation import _candidates
    root, manifest = original_bundle(spec)
    expected = spec["expected_original_universe_sha256"]
    entry = manifest["files"][manifest["candidate_universe_path"]]
    if expected != ORIGINAL_UNIVERSE_SHA or entry["sha256"] != expected:
        raise ValueError("OURS_REQUIRES_ORIGINAL66_NOT_REACH_OR_OTHER_PROPOSER")
    return _candidates(root, manifest)


def load_original_selector(root, manifest):
    """Original main B12 chooses among A1-A4; old GNN fixed-winner is not B12."""
    from src.ablations.gnn.cpu_evaluation import frozen_selector, _input
    from src.eval.mutagenicity_wnode_selector import VariantConfig, preregistered_variant_configs
    selector = frozen_selector(root, manifest)
    config = read_json(_input(root, manifest, "selector_variant_configs_path"))
    expected = {v.name: asdict(v) for v in preregistered_variant_configs()}
    if config["variants"] != expected:
        raise ValueError("ORIGINAL_B12_A1_A4_CONFIG_DRIFT")
    selector["variants"] = {name: VariantConfig(**values) for name, values in config["variants"].items()}
    selector["historical_selected_variant_not_adopted"] = selector.pop("variant").name
    selector["variant_decision_scope"] = "NEW_GIN_FIXED66_CALIBRATION_ONLY"
    return selector


def fixed_source_parents(spec: Mapping[str, Any], split: str, *, test_authorized=False):
    from src.ablations.gnn.cpu_training import bundle_file
    if split not in ("train", "calibration", "test"):
        raise ValueError("EXPERIMENT_SPLIT_UNSUPPORTED")
    if split == "test" and not test_authorized:
        raise ValueError("TEST_REQUIRES_CALLER_VERIFIED_NEW_SELECTOR_FREEZE")
    root, manifest = original_bundle(spec)
    parents = load_bace_parents(bundle_file(root, manifest, manifest["splits"][split]), source_label=1)
    if len({p.parent_id for p in parents}) != len(parents):
        raise ValueError("DUPLICATE_BASE_COHORT_PARENT")
    expected = {"train": 386, "calibration": 66, "test": 141}[split]
    if len(parents) != expected or any(p.label != 1 for p in parents):
        raise ValueError(f"FIXED_BASE_COHORT_CHANGED:{split}:{len(parents)}")
    return parents


def validate_gin_adoption(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Adopt accepted weights and actual validation fit, never refit or relabel."""
    root = Path(spec["gin_root"]).resolve(strict=True)
    adoption = spec["classifier_adoption"]
    source = Path(adoption["root"]).resolve(strict=True)
    docs = {}
    for role in ("acceptance", "independent_science_replay"):
        item = adoption[role]
        rel = Path(item["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("ADOPTION_PATH_ESCAPE")
        docs[role] = _bound_json(source / rel, item["sha256"])
    accepted, replay = docs["acceptance"], docs["independent_science_replay"]
    expected = spec["gin_files"]
    model = replay["models"]["gin"]
    if (accepted.get("state") != "GNN_CORE_SEED7_CORRECTED_PASS"
            or accepted.get("all_weights_unchanged") is not True
            or accepted.get("independent_science_replay_sha256") != adoption["independent_science_replay"]["sha256"]
            or replay.get("state") != "PASS"
            or replay.get("all_five_validation_temperatures_fitted_and_input_bound") is not True
            or model["model_sha256"] != expected["model.pt"]
            or model["temperature_sha256"] != expected["temperature_scaling.json"]
            or Path(model["root"]).resolve(strict=True) != root):
        raise ValueError("CORRECTED_GIN_ACCEPTANCE_OR_MODEL_BINDING_CONFLICT")
    temperature = _bound_json(root / "temperature_scaling.json", expected["temperature_scaling.json"])
    _bound_json(root / "feature_schema.json", expected["feature_schema.json"])
    overlay = read_json(root / "classifier_overlay_manifest.json")
    fit = _bound_json(root / "temperature_fit_receipt.json", overlay["fit_receipt_sha256"])
    value = temperature.get("temperature")
    if (type(value) not in (float, int) or not math.isfinite(value) or value <= 0
            or temperature.get("status") != "fit" or temperature.get("num_examples") != 187
            or temperature.get("selection_split") != "validation"
            or temperature.get("test_used_for_fit") is not False
            or fit.get("status") != "fit" or fit.get("fit_split") != "validation"
            or fit.get("num_examples") != 187 or fit.get("test_used") is not False
            or fit.get("calibration_used") is not False
            or fit.get("model_weight_sha") != expected["model.pt"]
            or fit.get("fitted_temperature") != value
            or fit.get("optimizer_parameter_count") != 1 or fit.get("model_optimizer_parameters") != 0
            or len(fit.get("sample_ids", [])) != 187 or len(set(fit["sample_ids"])) != 187
            or overlay.get("weights_changed") is not False
            or overlay.get("weight_sha256") != expected["model.pt"]
            or overlay.get("temperature_sha256") != expected["temperature_scaling.json"]):
        raise ValueError("GIN_ACTUAL_VALIDATION187_FIT_BINDING_REQUIRED")
    return dict(scope=SCOPE, backbone="gin", model_root=str(root), model_sha256=expected["model.pt"],
        temperature=value, temperature_sha256=expected["temperature_scaling.json"],
        feature_schema_sha256=expected["feature_schema.json"], validation_examples=187,
        original_acceptance=adoption["acceptance"], fit_receipt_sha256=overlay["fit_receipt_sha256"],
        weights_retrained=False, temperature_refitted=False, old_gin_selector_adopted=False,
        old_gine_flips_adopted=False, prior_test_evaluations_disclosed=True)


def build_runtime(spec: Mapping[str, Any], output: str | Path, *, split="train",
                  test_freeze=None, validate_test_freeze=None):
    """Build one CPU oracle/distance runtime; the caller owns stage/parent commits."""
    from src.ablations.gnn.cpu_evaluation import _featurizer, _distance
    from src.ablations.gnn.reach_raw_distance_reuse import raw_contract_from_bundle
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    from src.oracles.gnn_oracle import GNNOracle
    if split not in ("train", "calibration", "test"):
        raise ValueError("EXPERIMENT_SPLIT_UNSUPPORTED")
    if split == "test":
        if test_freeze is None or not callable(validate_test_freeze):
            raise ValueError("TEST_RUNTIME_REQUIRES_ACTUAL_NEW_FREEZE_VALIDATOR")
        validate_test_freeze(test_freeze)
    root, manifest = original_bundle(spec)
    out = Path(output).resolve()
    model_root = Path(spec["gin_root"]).resolve(strict=True)
    if any(out == p or p in out.parents or out in p.parents for p in (root, model_root)):
        raise ValueError("RUNTIME_OUTPUT_OVERLAPS_FROZEN_INPUT")
    adopted = validate_gin_adoption(spec)
    if manifest["files"][manifest["feature_schema_path"]]["sha256"] != adopted["feature_schema_sha256"]:
        raise ValueError("GIN_AND_SHARED_FEATURIZER_SCHEMA_DIFFER")
    # Loaded once per task. Existing bundle loader validates actual model bytes.
    oracle = GNNOracle.from_checkpoint(model_root, device="cpu", batch_size=int(spec.get("batch_size", 256)))
    if (oracle.backbone != "gin" or oracle.checkpoint_id != adopted["model_sha256"]
            or oracle.temperature != adopted["temperature"] or oracle.source_label != 1 or oracle.num_classes != 2):
        raise ValueError("ACTUAL_LOADED_GIN_DIFFERS_FROM_ADOPTION")
    for parameter in oracle.model.parameters():
        parameter.requires_grad_(False)
    oracle.model.eval()
    out.mkdir(parents=True, exist_ok=True)
    distance = _distance(root, manifest, out)
    install_compact_node_cache(distance)
    try:
        source_index = spec.get("raw_cost_indexes", {}).get(split)
        if split != "train" and source_index is None:
            raise ValueError("SEALED_RAW_COST_ADOPTION_REQUIRED_BEFORE_EVALUATION")
        if source_index is not None:
            index = _bound_json(Path(source_index["path"]), source_index["sha256"])
            if index.get("split") != split:
                raise ValueError("RAW_COST_SPLIT_BINDING_CONFLICT")
            if split == "test":
                expected_freeze = source_index.get("new_test_freeze_sha256")
                if not expected_freeze or index.get("new_test_freeze_sha256") != expected_freeze:
                    raise ValueError("RAW_COST_TEST_FREEZE_BINDING_CONFLICT")
                # The caller's validator owns the new experiment's freeze schema;
                # this adapter also verifies the index was migrated for that file.
                freeze_path = Path(source_index["new_test_freeze_path"])
                if sha256_file(freeze_path) != expected_freeze or read_json(freeze_path) != test_freeze:
                    raise ValueError("RAW_COST_BELONGS_TO_DIFFERENT_EXPERIMENT_FREEZE")
            distance = with_native_graph_distance(distance, index=index,
                current_raw_contract=raw_contract_from_bundle(manifest), repo=Path(__file__).resolve().parents[2])
        receipt = dict(adopted, loaded_model_eval=True, loaded_trainable_parameters=0,
            total_parameters=sum(p.numel() for p in oracle.model.parameters()),
            split=split, device="cpu", main_matrix_write=False)
        atomic_json(out / "gin_adoption_receipt.json", receipt)
        return dict(oracle=oracle, featurizer=_featurizer(root, manifest), distance=distance,
            selector=load_original_selector(root, manifest), adoption_receipt=receipt)
    except BaseException:
        distance.close()
        raise


def evaluate_parent(parent, candidates, oracle, featurizer, distance, split, *,
                    batch_size=256, parent_prediction=None):
    """Original connected hard-deletion all-match semantics and GIN own minimum."""
    from src.eval.bace_frozen_gnn_verification import _evaluate_rows
    if parent.label != 1 or oracle.backbone != "gin" or oracle.source_label != 1:
        raise ValueError("FROZEN_GIN_SOURCE1_REQUIRED")
    if split not in ("train", "calibration", "test"):
        raise ValueError("EXPERIMENT_SPLIT_UNSUPPORTED")
    cache = None
    if parent_prediction is not None:
        if (parent_prediction.get("checkpoint_id") != oracle.checkpoint_id
                or parent_prediction.get("backbone") != "gin"
                or parent_prediction.get("temperature") != oracle.temperature):
            raise ValueError("PARENT_PREDICTION_CACHE_NOT_BOUND_TO_CURRENT_GIN")
        cache = {parent.parent_id: dict(parent_smiles=parent.smiles,
            pred_before=parent_prediction["predicted_label"], p_before=parent_prediction["probabilities"])}
    pairs, matches = _evaluate_rows([parent], candidates, oracle=oracle, featurizer=featurizer,
        distance_provider=distance, oracle_batch_size=batch_size, split=split,
        oracle_checkpoint_id=oracle.checkpoint_id, parent_prediction_cache=cache)
    if any(row.get("cf_flip") and row.get("distance_ok") is not True for row in matches):
        raise ValueError("STRICT_FLIP_RAW_DISTANCE_FAILURE_NOT_ZERO_COVERAGE")
    if len(pairs) != len(candidates):
        raise ValueError("ORIGINAL_POOL_PAIR_COVERAGE_INCOMPLETE")
    for row in (*pairs, *matches):
        row.update(split=split, oracle_backbone="gin", oracle_temperature=oracle.temperature)
    return pairs, matches


def select_calibration(matrix, selector):
    """Replay the main B12's four variants and its calibration-only decision."""
    from src.ablations.gnn.cpu_evaluation import select_calibration as original_variant
    from src.eval.mutagenicity_wnode_selector import (
        VARIANT_NAMES, build_candidate_chemistry, build_coverage_redundancy_matrix,
        compute_prefix_metrics, build_variant_comparison_row, choose_variant,
    )
    if (matrix.manifest.get("split") != "calibration"
            or matrix.manifest.get("test_loaded") is not False):
        raise ValueError("ORIGINAL_B12_VARIANT_DECISION_CALIBRATION_ONLY")
    if set(selector.get("variants", {})) != set(VARIANT_NAMES):
        raise ValueError("ORIGINAL_B12_REQUIRES_ALL_FOUR_VARIANTS_NOT_OLD_WINNER")
    chemistry = build_candidate_chemistry(matrix.candidate_rows, size_normalization_rows=matrix.full_candidate_rows)
    redundancy = build_coverage_redundancy_matrix(matrix.distances, selector["thresholds"].levels)
    sequences, traces, comparison = {}, {}, []
    for name in VARIANT_NAMES:
        variant = selector["variants"][name]
        active = dict(selector, variant=variant)
        sequence, trace = original_variant(matrix, active)
        metrics, _ = compute_prefix_metrics(sequence, matrix=matrix, thresholds=selector["thresholds"],
            coverage_redundancy_matrix=redundancy, structural_similarity_matrix=chemistry.structural_similarity)
        comparison.append(build_variant_comparison_row(variant, metrics, table_k=10, top_k=20,
            prefix_weights=selector["prefix_weights"], final_objective=trace["objective"]))
        sequences[name], traces[name] = sequence, trace
    decision = choose_variant(comparison)
    return sequences[decision["variant"]], dict(selected_variant=decision["variant"],
        selected_metrics=decision, variant_comparison=comparison, variant_traces=traces,
        variant_sequences=sequences, original_b12_a1_a4_replayed=True,
        historical_selected_variant_adopted=False, test_used=False,
        original_thresholds_refitted=False, original_selector_input_sha256=selector["input_sha256"])


def train_only_timing(spec: Mapping[str, Any], output: str | Path):
    """Two predetermined train parents and all66 rules; never a selector fit."""
    import resource
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidates = load_original_pool(spec)
    parents = fixed_source_parents(spec, "train")[:2]
    started = time.monotonic()
    runtime = build_runtime(spec, out / "runtime", split="train")
    load_seconds = time.monotonic() - started
    try:
        rows = []
        for parent in parents:
            begin = time.monotonic()
            pairs, matches = evaluate_parent(parent, candidates, split="train",
                **{key: runtime[key] for key in ("oracle", "featurizer", "distance")},
                batch_size=int(spec.get("batch_size", 256)))
            elapsed = time.monotonic() - begin
            science = dict(pair_rows=pairs, match_rows=matches)
            atomic_json(out / f"parent-{parent.parent_id}.json", dict(parent=asdict(parent),
                science=science, science_sha256=stable_sha256(science), elapsed_seconds=elapsed))
            rows.append(dict(parent_id=parent.parent_id, seconds=elapsed, pairs=len(pairs),
                matches=len(matches), strict_flip_pairs=sum(bool(p["pair_strict_flip"]) for p in pairs),
                pred_before=pairs[0]["pred_before"]))
        receipt = dict(scope=SCOPE, state="TRAIN_ONLY_TIMING_COMPLETE_NOT_EXPERIMENT_PASS",
            parent_ids=[p.parent_id for p in parents], candidate_count=66, parent_count=2,
            load_seconds=load_seconds, elapsed_seconds=time.monotonic()-started, parents=rows,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            distance_statistics=runtime["distance"].stats_dict(), calibration_loaded=False,
            test_loaded=False, selector_fitted=False, model_trained=False, temperature_refitted=False,
            main_matrix_write=False, resource_only_extrapolation=True,
            adoption=runtime["adoption_receipt"])
        atomic_json(out / "timing.json", receipt)
        return receipt
    finally:
        runtime["distance"].close()
