"""BACE-only fixed-pool cross-classifier evaluation, never a main publisher.

Parent checkpoints and existing evaluator/selector adapters are reused. This
module contains no generator, trainer, GPU scheduler or main-matrix authority.
"""
from __future__ import annotations

import csv
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.eval.bace_frozen_gnn_contracts import (
    atomic_csv, atomic_json, atomic_jsonl, read_json, read_jsonl,
    sha256_file, stable_sha256, utc_now,
)

EXPERIMENT = "BACE_GIN_FOUR_METHODS_FIXED_POOL_V1"
METHODS = ("ours", "globalgce", "gcfexplainer", "comrecgc")
LABELS = dict(zip(METHODS, ("Ours", "GlobalGCE", "GCFExplainer", "ComRecGC")))


def bound_json(binding: Mapping[str, Any]) -> dict:
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"INPUT_HASH_MISMATCH:{path}")
    return read_json(path)


def validate_spec(spec: Mapping[str, Any]) -> None:
    required_false = ("training_rerun", "temperature_refit", "candidate_generation_repeated",
                      "reach_v2_candidates_used", "main_matrix_write")
    if spec.get("experiment_id") != EXPERIMENT or any(spec.get(k) is not False for k in required_false):
        raise ValueError("EXPERIMENT_SCOPE_MISMATCH")
    if set(spec.get("pools", {})) != set(METHODS):
        raise ValueError("FOUR_METHOD_POOL_INVENTORY_REQUIRED")
    if spec.get("source_class") != 1 or spec.get("destination_class") != 0:
        raise ValueError("SOURCE_DESTINATION_CHANGED")
    if spec.get("base_counts") != {"calibration": 66, "test": 141}:
        raise ValueError("FIXED_BASE_COHORT_CONTRACT_CHANGED")
    if spec.get("test_results_previously_observed") is not True:
        raise ValueError("POST_HOC_DISCLOSURE_REQUIRED")
    if spec.get("rule_budget_semantics") != "AT_MOST_K":
        raise ValueError("AT_MOST_K_REQUIRED")
    output = Path(spec["output_root"]).resolve()
    allowed = ("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments/",
               "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiments/",
               "/private/tmp/")
    if not any(str(output).startswith(p) for p in allowed):
        raise ValueError("EXPERIMENT_OUTPUT_OUTSIDE_SCOPE")


def pool(spec: Mapping[str, Any], method: str) -> tuple[dict, list[dict]]:
    manifest = bound_json(spec["pools"][method])
    if manifest.get("method_id") != method or manifest.get("candidate_generation_repeated") is not False:
        raise ValueError("METHOD_SPECIFIC_ORIGINAL_POOL_REQUIRED")
    if manifest.get("reach_v2_candidates_used") is not False:
        raise ValueError("REACH_POOL_FORBIDDEN")
    if manifest.get("state") == "BLOCKED_MATERIALIZATION":
        return manifest, []
    source = Path(manifest["candidate_file"])
    # Only this small/newly imported candidate input is checked, never old bundles.
    if sha256_file(source) != manifest["pool_content_sha256"]:
        raise ValueError("CANDIDATE_CONTENT_CHANGED")
    rows = read_jsonl(source)
    ids = [str(r["candidate_id"]) for r in rows]
    if len(ids) != manifest["input_count"] or len(ids) != len(set(ids)):
        raise ValueError("POOL_COUNT_OR_ID_MISMATCH")
    return manifest, rows


def freeze_path(spec: Mapping[str, Any], method: str) -> Path:
    return Path(spec["output_root"]) / method / "selection_freeze.json"


def verify_freeze(spec: Mapping[str, Any], method: str) -> dict:
    receipt = read_json(freeze_path(spec, method))
    if (receipt.get("state") != "FROZEN" or receipt.get("test_loaded") is not False
            or receipt.get("spec_sha256") != stable_sha256(spec)
            or receipt.get("pool_manifest_sha256") != spec["pools"][method]["sha256"]):
        raise ValueError("NEW_GIN_CALIBRATION_FREEZE_REQUIRED")
    ids = receipt["ordered_rule_ids"]
    if len(ids) > 20 or len(ids) != len(set(ids)) or receipt["order_sha256"] != stable_sha256(ids):
        raise ValueError("FROZEN_RULE_ORDER_INVALID")
    if receipt.get("oracle_backbone") != "gin" or receipt.get("old_gine_flip_masks_reused") is not False:
        raise ValueError("OLD_CLASSIFIER_DECISIONS_FORBIDDEN")
    return receipt


def plan(spec: Mapping[str, Any]) -> dict:
    validate_spec(spec)
    root = Path(spec["output_root"])
    manifests = {m: pool(spec, m)[0] for m in METHODS}
    thresholds = bound_json(spec["thresholds"])
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    bundle = threshold_bundle_from_dict(thresholds)
    if bundle.theta_star != 0.008413518173529859 or bundle.cost_cap != 0.02956508038627219:
        raise ValueError("ORIGINAL_BACE_THRESHOLD_CONTRACT_CHANGED")
    from src.experiments.bace_gin_ours import fixed_source_parents, validate_gin_adoption, original_bundle
    adopted = validate_gin_adoption(spec)
    _, original = original_bundle(spec)
    parents = fixed_source_parents(spec, "calibration")
    ids = [p.parent_id for p in parents]
    contract = {"experiment_id": EXPERIMENT, "state": "PLANNED", "spec_sha256": stable_sha256(spec),
        "calibration_parent_ids": ids, "calibration_parent_ids_sha256": stable_sha256(ids),
        "test_manifest_binding": {"relative_path":original["splits"]["test"],
            **original["files"][original["splits"]["test"]]},
        "test_payload_opened": False, "base_counts": spec["base_counts"],
        "thresholds": thresholds, "pools": manifests,
        "scope": "POST_HOC_FIXED_POOL_CROSS_CLASSIFIER_COMPARISON",
        "test_results_previously_observed": True, "model_choice_inspired_by_observed_backbone_results": True,
        "main_matrix_write": False, "created_at": utc_now()}
    path = root / "manifests" / "experiment_contract.json"
    if path.exists():
        old = read_json(path)
        if old["spec_sha256"] != stable_sha256(spec):
            raise ValueError("FRESH_ROOT_REQUIRED_FOR_CHANGED_SPEC")
        return old
    atomic_json(path, contract)
    atomic_json(root / "manifests" / "oracle_contract.json", adopted)
    atomic_json(root / "manifests" / "candidate_pool_manifest.json", manifests)
    atomic_json(root / "manifests" / "cohort_manifest.json", {k: contract[k] for k in (
        "calibration_parent_ids", "calibration_parent_ids_sha256", "base_counts", "test_payload_opened")})
    return contract


def evaluate(spec: Mapping[str, Any], method: str, split: str, start: int, stop: int | None) -> dict:
    """Run one bounded stable parent range; resume skips sealed complete units."""
    validate_spec(spec)
    root = Path(spec["output_root"])
    contract = read_json(root / "manifests" / "experiment_contract.json")
    if contract["spec_sha256"] != stable_sha256(spec):
        raise ValueError("PLAN_BINDING_MISMATCH")
    if split not in ("calibration", "test"):
        raise ValueError("INVALID_SCIENCE_SPLIT")
    frozen = verify_freeze(spec, method) if split == "test" else None
    manifest, candidates = pool(spec, method)
    if manifest.get("state") == "BLOCKED_MATERIALIZATION":
        return {"method": method, "state": "BLOCKED_MATERIALIZATION", "reason": manifest.get("reason")}
    if frozen is not None:
        indexed = {r["candidate_id"]: r for r in candidates}
        candidates = [indexed[x] for x in frozen["ordered_rule_ids"]]
    from src.experiments import bace_gin_ours as ours
    runtime_spec = copy.deepcopy(spec)
    if method in ("gcfexplainer", "comrecgc"):
        # Native original full-graph costs need their own provenance migration.
        # The sparse Ours deletion index is not proof all native costs are absent.
        path = root / "manifests" / f"{method}_native_raw_adoption.json"
        if not path.is_file():
            raise ValueError(f"NATIVE_RAW_DISTANCE_ADOPTION_REQUIRED:{method}:{path}")
        adoption = read_json(path)
        if (adoption.get("spec_sha256") != stable_sha256(spec)
                or adoption.get("pool_manifest_sha256") != spec["pools"][method]["sha256"]):
            raise ValueError("NATIVE_RAW_ADOPTION_WRONG_EXPERIMENT")
        source_index = adoption.get(split)
        if source_index is None:
            raise ValueError(f"NATIVE_RAW_SPLIT_ADOPTION_MISSING:{method}:{split}")
        runtime_spec.setdefault("raw_cost_indexes",{})[split] = source_index
    if split == "test" and method == "ours":
        from src.ablations.gnn.reach_raw_distance_reuse import build_index
        index_path = root / method / "raw_test_adoption.json"
        fp = freeze_path(spec, method)
        build_index(spec["raw_distance_source"], split="test", output=index_path,
            repo=Path(__file__).resolve().parents[2], test_freeze_path=fp,
            test_freeze_sha=sha256_file(fp), validate_test_freeze=lambda _: verify_freeze(spec,method))
        runtime_spec.setdefault("raw_cost_indexes", {})["test"] = {"path": str(index_path),
            "sha256":sha256_file(index_path),"new_test_freeze_path":str(fp),"new_test_freeze_sha256":sha256_file(fp)}
    runtime = ours.build_runtime(runtime_spec, root / method / "runtime" / split / f"range-{start}",
        split=split, test_freeze=frozen,
        validate_test_freeze=lambda *args, **kwargs: verify_freeze(spec, method))
    parents = ours.fixed_source_parents(spec, split, test_authorized=frozen is not None)
    if len(parents) != spec["base_counts"][split]:
        raise ValueError("BASE_COHORT_COUNT_MISMATCH")
    end = len(parents) if stop is None else min(stop, len(parents))
    if not 0 <= start < end:
        raise ValueError("INVALID_PARENT_RANGE")
    oracle_binding = {"backbone": "gin", "model_sha256": spec["gin_files"]["model.pt"],
        "temperature_sha256": spec["gin_files"]["temperature_scaling.json"],
        "feature_schema_sha256": spec["gin_files"]["feature_schema.json"]}
    predictions = None
    if method in ("gcfexplainer", "comrecgc"):
        from src.experiments import bace_gin_native_baselines as native
        native.validate_original_pool(method, pool(spec, method)[1])
        # One cache per parent range avoids concurrent writers. Prediction only,
        # no old classifier masks/distances; complete graphs predicted in batches.
        predictions = native.prepare_candidate_predictions(candidates, runtime["oracle"],
            runtime["featurizer"], oracle_binding=oracle_binding, batch_size=spec.get("batch_size", 64))
    completed = 0
    started = time.monotonic()
    for i in range(start, end):
        parent = parents[i]
        out = root / method / split / f"parent-{i:05d}"
        out.mkdir(parents=True, exist_ok=True)
        binding = stable_sha256({"spec": stable_sha256(spec), "method": method, "split": split,
            "parent_id": parent.parent_id, "candidates": [r["candidate_id"] for r in candidates],
            "freeze": frozen})
        with (out / "writer.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            checkpoint = out / "complete.json"
            if checkpoint.exists():
                old = read_json(checkpoint)
                if old["binding"] != binding:
                    raise ValueError("PARENT_CHECKPOINT_BINDING_MISMATCH")
                completed += 1
                continue
            if method == "ours":
                pairs, matches = ours.evaluate_parent(parent, candidates, runtime["oracle"],
                    runtime["featurizer"], runtime["distance"], split, batch_size=spec.get("batch_size", 64))
            elif method in ("gcfexplainer", "comrecgc"):
                pairs, matches = native.evaluate_parent(parent, candidates, runtime["oracle"],
                    runtime["featurizer"], runtime["distance"], split, method=method,
                    oracle_binding=oracle_binding, candidate_predictions=predictions)
            else:
                raise ValueError("GLOBALGCE_MATERIALIZATION_EVALUATOR_NOT_YET_ADMITTED")
            expected = {(parent.parent_id, str(c["candidate_id"])) for c in candidates}
            if len(pairs) != len(expected) or {(str(p["parent_id"]),str(p["candidate_id"])) for p in pairs} != expected:
                raise ValueError("PARENT_CANDIDATE_COVERAGE_MISMATCH")
            atomic_jsonl(out / "pairs.jsonl", pairs)
            atomic_jsonl(out / "applications.jsonl", matches)
            atomic_json(checkpoint, {"state": "COMPLETE", "binding": binding, "parent_id": parent.parent_id,
                "parent_index": i, "candidate_count": len(candidates), "spec_sha256":stable_sha256(spec),
                "pairs_sha256": sha256_file(out / "pairs.jsonl"),
                "completed_at": utc_now(), "oracle_backbone": "gin"})
            completed += 1
    result = {"state": "PARENT_RANGE_COMPLETE", "method": method, "split": split,
        "start": start, "stop": end, "completed_units": completed, "elapsed_seconds": time.monotonic() - started}
    result["raw_distance_statistics"] = runtime["distance"].stats_dict()
    result["raw_distance_reuse"] = getattr(runtime["distance"], "used", [])
    runtime["distance"].close()
    atomic_json(root / method / split / f"range-{start:05d}-{end:05d}.json", result)
    return result


def _rows(spec: Mapping[str, Any], method: str, split: str):
    count = spec["base_counts"][split]
    for i in range(count):
        parent_root = Path(spec["output_root"]) / method / split / f"parent-{i:05d}"
        receipt = read_json(parent_root / "complete.json")
        if (receipt["state"] != "COMPLETE" or receipt["parent_index"] != i
                or receipt.get("spec_sha256") != stable_sha256(spec)):
            raise ValueError("INCOMPLETE_PARENT_PARTITION")
        path = parent_root / "pairs.jsonl"
        if sha256_file(path) != receipt["pairs_sha256"]:
            raise ValueError("SEALED_PARENT_PAIR_CHANGED")
        yield from read_jsonl(path)


def freeze(spec: Mapping[str, Any], method: str) -> dict:
    validate_spec(spec)
    if freeze_path(spec, method).exists():
        return verify_freeze(spec, method)
    manifest, candidates = pool(spec, method)
    if manifest.get("state") == "BLOCKED_MATERIALIZATION":
        raise ValueError("BLOCKED_MATERIALIZATION_NOT_VALID_ZERO")
    root = Path(spec["output_root"]) / method
    cal = root / "calibration_matrix"
    cal.mkdir(parents=True, exist_ok=True)
    # Stream the matrix; GCF has 1.45M rows, not a list of large Python dicts.
    temporary = cal / "pair_matrix.jsonl.partial"
    parents, n, flips = [], 0, 0
    with temporary.open("w") as out:
        for row in _rows(spec, method, "calibration"):
            if row["parent_id"] not in parents:
                parents.append(row["parent_id"])
            if row.get("split") != "calibration":
                raise ValueError("SELECTOR_TEST_LEAKAGE")
            out.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
            flips += bool(row["pair_strict_flip"])
        out.flush()
        os.fsync(out.fileno())
    if len(parents) != 66 or n != 66 * len(candidates):
        raise ValueError("INCOMPLETE_FULL_CALIBRATION_CARTESIAN_PRODUCT")
    os.replace(temporary, cal / "pair_matrix.jsonl")
    atomic_jsonl(cal / "selected_candidate_universe.jsonl", candidates)
    atomic_json(cal / "summary.json", {"parent_count": 66, "selected_candidate_count": len(candidates),
        "strict_flip_pair_count": flips, "test_loaded": False})
    atomic_json(cal / "run_manifest.json", {"inputs": {"cohort_name": "calibration"},
        "split": "calibration", "test_loaded": False, "classifier_family": "gin"})
    from src.eval.mutagenicity_wnode_selector import load_calibration_matrix
    if method == "ours":
        # B12 originally compared all four preregistered variants on calibration.
        # The GNN sensitivity adapter's fixed old winner is NOT that decision.
        from src.eval.mutagenicity_wnode_selector import run_mutagenicity_wnode_selector, threshold_bundle_from_dict
        from src.experiments.bace_gin_ours import original_bundle
        from src.ablations.gnn.cpu_training import bundle_file
        bundle_root, bundle = original_bundle(spec)
        config = read_json(bundle_file(bundle_root,bundle,bundle["selector_variant_configs_path"]))
        if config["top_k"] != 20 or config["table_k"] != 10:
            raise ValueError("ORIGINAL_B12_BUDGET_CHANGED")
        run_mutagenicity_wnode_selector(matrix_run_dir=cal,output_dir=root/"selector",
            top_k=20,table_k=10,seed=13,local_swap_passes=config["local_swap_passes"],
            prefix_weights=config["prefix_weights"],candidate_limit=0,parent_limit=0,forbid_test=True,
            frozen_thresholds=threshold_bundle_from_dict(bound_json(spec["thresholds"])),
            frozen_threshold_provenance=manifest["threshold_provenance"])
        decision=read_json(root/"selector"/"calibration_decision.json")
        selected=read_json(root/"selector"/"variants"/decision["selected_variant"]/"selected_top20.json")
        ids=[r["candidate_id"] for r in selected["candidates"]]
        details={"selected_variant":decision["selected_variant"],"decision_rule":decision["decision_rule"],
            "four_original_variants_replayed":True,"old_gine_winner_reused":False}
    else:
        from src.experiments.bace_gin_native_baselines import select_order
        details = select_order(cal, {"method": method, "test_loaded": False,
            "original_selector_config": manifest["original_selector_config"],
            "thresholds": bound_json(spec["thresholds"]),
            "threshold_provenance": manifest["threshold_provenance"],
            "output_root": str(root / "selector")})
        ids = details["ordered_rule_ids"]
    receipt = {"state": "FROZEN", "experiment_id": EXPERIMENT, "method": method,
        "spec_sha256": stable_sha256(spec), "pool_manifest_sha256": spec["pools"][method]["sha256"],
        "ordered_rule_ids": ids, "order_sha256": stable_sha256(ids), "test_loaded": False,
        "oracle_backbone": "gin", "old_gine_flip_masks_reused": False,
        "calibration_matrix_sha256": sha256_file(cal / "pair_matrix.jsonl"),
        "selector_details": details, "created_at": utc_now()}
    atomic_json(freeze_path(spec, method), receipt)
    return receipt


def prefix_metrics(parent_ids: Sequence[str], candidates: Sequence[str], rows: Sequence[Mapping[str, Any]],
                   *, theta: float, cap: float, endpoints: Sequence[float]) -> dict:
    """Exact AT_MOST_K metrics; fixed base and native share one frozen sequence."""
    if len(candidates) > 20 or len(candidates) != len(set(candidates)):
        raise ValueError("INVALID_FROZEN_PREFIX")
    if not candidates:
        raise ValueError("BLOCKED_EMPTY_SELECTION_REQUIRES_EXPLICIT_PARENT_PREDICTIONS")
    by_pair = {(r["parent_id"], r["candidate_id"]): r for r in rows}
    if len(by_pair) != len(rows) or set(by_pair) != {(p,c) for p in parent_ids for c in candidates}:
        raise ValueError("TEST_CARTESIAN_PRODUCT_INCOMPLETE")
    before = {}
    for r in rows:
        p = r["parent_id"]
        if p in before and before[p] != int(r["pred_before"]):
            raise ValueError("INCONSISTENT_PARENT_GIN_PREDICTION")
        before[p] = int(r["pred_before"])
        if r["pair_strict_flip"] and (int(r["pred_before"]) != 1 or int(r["pred_after"]) != 0):
            raise ValueError("OLD_OR_WRONG_STRICT_FLIP_MASK")
    native = [p for p in parent_ids if before[p] == 1]
    output, distances, ecdf = [], [], []
    for k in range(1, 21):
        prefix = candidates[:k]
        best = {}
        for p in parent_ids:
            valid = [float(by_pair[p,c]["wnode_distance"]) for c in prefix if by_pair[p,c]["pair_strict_flip"]]
            if any(not math.isfinite(v) or v < 0 for v in valid):
                raise ValueError("RAW_DISTANCE_INVALID")
            best[p] = min(valid, default=math.inf)
            distances.append({"parent_id":p,"K_requested":k,"K_effective":len(prefix),
                "pred_before":before[p],"best_valid_distance":best[p] if math.isfinite(best[p]) else None})
        for cohort, ids in (("fixed141", parent_ids), ("gin_native", native)):
            values = [best[p] for p in ids]
            finite = sorted(v for v in values if math.isfinite(v))
            n = len(ids)
            median = (finite[(len(finite)-1)//2] + finite[len(finite)//2])/2 if finite else None
            output.append({"cohort":cohort,"denominator":n,"K_requested":k,"K_effective":len(prefix),
                "covered_count":sum(v <= theta for v in values),
                "coverage":sum(v <= theta for v in values)/n if n else None,
                "finite_strict_flip_count":len(finite),"strict_flip_availability":len(finite)/n if n else None,
                "fixed_capped_mean":sum(min(v,cap) for v in values)/n if n else None,
                "conditional_median":median,"theta_star":theta,"cost_cap":cap})
            if k in (10,20):
                upper = max(endpoints)
                xs = sorted({0.,theta,*map(float,endpoints),*(v for v in finite if v <= upper)})
                ecdf.extend({"cohort":cohort,"denominator":n,"K_requested":k,"K_effective":len(prefix),
                    "threshold":x,"covered_count":sum(v<=x for v in values),
                    "coverage":sum(v<=x for v in values)/n if n else None} for x in xs)
    return {"prefix_rows":output,"parent_distances":distances,"exact_ecdf":ecdf,
        "parent_predictions":[{"parent_id":p,"pred_before":before[p],"in_native":p in native} for p in parent_ids]}


def aggregate(spec: Mapping[str, Any], method: str) -> dict:
    frozen = verify_freeze(spec, method)
    rows = list(_rows(spec, method, "test"))
    parents = list(dict.fromkeys(r["parent_id"] for r in rows))
    if len(parents) != 141:
        raise ValueError("FIXED141_DENOMINATOR_CHANGED")
    t = bound_json(spec["thresholds"])
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    th = threshold_bundle_from_dict(t)
    result = prefix_metrics(parents, frozen["ordered_rule_ids"], rows, theta=th.theta_star,
                            cap=th.cost_cap, endpoints=th.raw_thresholds)
    result.update(state="EVALUATED", method=method, spec_sha256=stable_sha256(spec),
        freeze_sha256=sha256_file(freeze_path(spec, method)), main_matrix_write=False)
    result["failure_funnel"] = {"split":"test", "pool_scope":"new_calibration_selected_prefix",
        "base_parent_count":len(parents),"selected_rule_count":len(frozen["ordered_rule_ids"]),
        "pair_count":len(rows),"gin_source_parents":sum(r["in_native"] for r in result["parent_predictions"]),
        "applicable_pairs":sum(bool(r["applicable"]) for r in rows),
        "strict_flip_pairs_with_exact_distance":sum(bool(r["pair_strict_flip"]) for r in rows),
        "unavailable_distance_not_relabelled_zero":True}
    atomic_json(Path(spec["output_root"]) / method / "metrics.json", result)
    return {k:v for k,v in result.items() if not isinstance(v,list)}


def export(spec: Mapping[str, Any]) -> dict:
    root = Path(spec["output_root"])
    ready, pending, table, f3, f4, native, parents, distances, funnels = [], [], [], [], [], [], [], [], []
    for method in METHODS:
        path = root / method / "metrics.json"
        if not path.exists():
            pending.append(method)
            state = pool(spec,method)[0].get("state", "PENDING")
            table.append({"method":LABELS[method],"state":state,"cohort":"fixed141"})
            continue
        result = read_json(path)
        if result["spec_sha256"] != stable_sha256(spec) or result["freeze_sha256"] != sha256_file(freeze_path(spec,method)):
            raise ValueError("METRIC_LINEAGE_CHANGED")
        ready.append(method)
        def add(rows): return [{"method":LABELS[method], **r} for r in rows]
        f3 += add([r for r in result["prefix_rows"] if r["cohort"] == "fixed141"])
        native += add([r for r in result["prefix_rows"] if r["cohort"] == "gin_native"])
        table += add([{**r,"state":"EVALUATED"} for r in result["prefix_rows"] if r["cohort"]=="fixed141" and r["K_requested"]==10])
        f4 += add([r for r in result["exact_ecdf"] if r["cohort"]=="fixed141"])
        parents += add(result["parent_predictions"])
        distances += add(result["parent_distances"])
        funnels += add([result["failure_funnel"]])
    for name, rows in (("bace_gin_fixed141_table2",table),("bace_gin_fixed141_figure3",f3),
        ("bace_gin_fixed141_figure4_exact_ecdf",f4),("bace_gin_native_metrics",native),
        ("bace_gin_parent_predictions",parents),("bace_gin_parent_best_distances",distances),
        ("bace_gin_method_failure_funnel",funnels)):
        if rows: atomic_csv(root / "source_csv" / f"{name}.csv",rows)
    receipt = {"state":"PARTIAL" if pending else "EVALUATED_AWAITING_INDEPENDENT_AUDIT", "ready_methods":ready,
        "pending_methods":pending,"experiment_id":EXPERIMENT,"main_matrix_write":False,
        "scope":"METHOD_SPECIFIC_FIXED_POOL_CROSS_CLASSIFIER", "created_at":utc_now()}
    atomic_json(root / "experiment_registry.json",receipt)
    return receipt


def status(spec: Mapping[str, Any]) -> dict:
    root = Path(spec["output_root"])
    methods = {}
    for method in METHODS:
        m = {"calibration_units":len(list((root/method/"calibration").glob("parent-*/complete.json"))),
             "test_units":len(list((root/method/"test").glob("parent-*/complete.json"))),
             "selector_frozen":freeze_path(spec,method).is_file(), "metrics_present":(root/method/"metrics.json").is_file()}
        m["state"] = "EVALUATED" if m["metrics_present"] else "PARTIAL_OR_PENDING"
        methods[method]=m
    return {"experiment_id":EXPERIMENT,"snapshot_time":utc_now(),"methods":methods,"main_matrix_write":False}
