"""BACE A+ continuation using existing GIN, hard-deletion and raw-OT adapters.

No trainer, general scheduler or main matrix publisher. Successful full-parent
units are immutable and reused on resume; test opens only after global freeze.
"""
from __future__ import annotations

import copy
import fcntl
import math
import os
from pathlib import Path
import resource
import time

from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, atomic_csv, read_json, read_jsonl, sha256_file, stable_sha256, utc_now,
)
from src.experiments import bace_gin_ours as adapter
from src.experiments.bace_gin_fixed_pool import bound_json, prefix_metrics
from src.experiments.bace_gin_reach_selector import POLICY, select

EXPERIMENT = "BACE_GIN_REACH_ALIGNED_V2"


def seal(path, data):
    value = dict(data, self_sha256=stable_sha256(data))
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError("IMMUTABLE_A_PLUS_RECEIPT_CONFLICT:" + str(path))
    else:
        atomic_json(path, value)
    return value


def verified(path):
    data = read_json(path)
    if data.get("self_sha256") != stable_sha256({k: v for k, v in data.items() if k != "self_sha256"}):
        raise ValueError("A_PLUS_RECEIPT_BINDING_CONFLICT:" + str(path))
    return data


def validate(spec):
    if (spec["experiment_id"] != EXPERIMENT or spec["main_matrix_write"] is not False
            or spec["training_rerun"] is not False or spec["temperature_refit"] is not False
            or spec["additional_ppo_updates"] != 0 or spec["selector_policy"] != POLICY
            or spec["base_counts"] != {"train": 386, "calibration": 66, "test": 141}
            or spec["test_results_previously_observed"] is not True):
        raise ValueError("A_PLUS_SCIENTIFIC_SCOPE_DRIFT")
    target = Path(spec["output_root"]).resolve()
    if not str(target).startswith("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments/bace-gin-reach-aligned-v2/"):
        raise ValueError("A_PLUS_HPC_OUTPUT_SCOPE")


def pools(spec):
    from src.ablations.gnn.reach_v2_adapter import validate_pool
    old = adapter.load_original_pool(spec)
    frozen = verified(spec["reach_candidate_freeze"])
    search = verified(spec["reach_search_contract"])
    path = Path(spec["reach_candidate_file"])
    if frozen["search_contract_sha256"] != search["self_sha256"]:
        raise ValueError("REACH_TRAIN_SOURCE_CONTRACT_CONFLICT")
    digest = sha256_file(path)  # one newly adopted candidate file, not model bundles
    new = read_jsonl(path)
    validate_pool(new, frozen, current_pool_sha=digest, old_ids=[r["candidate_id"] for r in old])
    by_id = {r["candidate_id"]: r for r in new}
    for row in old:
        if by_id[row["candidate_id"]] != row:
            # Legacy retention may add train support provenance, never graph edits.
            if any(by_id[row["candidate_id"]].get(k) != v for k, v in row.items()):
                raise ValueError("ORIGINAL66_CONTENT_CHANGED")
    if frozen.get("actual_query_budget_exceeded") is not False or search.get("test_opened") is not False:
        raise ValueError("REACH_SOURCE_NOT_AUTHORIZED_TRAIN_ONLY")
    return old, new, frozen


def plan(spec):
    validate(spec)
    root = Path(spec["output_root"])
    if (root / "contract.json").exists():
        old = verified(root / "contract.json")
        if old["spec_sha256"] != stable_sha256(spec):
            raise ValueError("A_PLUS_SPEC_CHANGED")
        return old
    old, pool, freeze = pools(spec)
    adopted = adapter.validate_gin_adoption(spec)
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    thresholds = threshold_bundle_from_dict(bound_json(spec["thresholds"]))
    if thresholds.theta_star != .008413518173529859 or thresholds.cost_cap != .02956508038627219:
        raise ValueError("BACE_FROZEN_THRESHOLDS_CHANGED")
    old_spec = bound_json(spec["v1_spec"])
    old_freeze = read_json(Path(old_spec["output_root"]) / "ours/selection_freeze.json")
    if old_freeze["spec_sha256"] != stable_sha256(old_spec) or old_freeze["test_loaded"] is not False:
        raise ValueError("V1_CALIBRATION_FREEZE_INVALID")
    from src.experiments.bace_gin_fixed_pool import verify_freeze
    verify_freeze(old_spec, "ours")
    root.mkdir(parents=True, exist_ok=True)
    return seal(root / "contract.json", dict(experiment_id=EXPERIMENT,
        spec_sha256=stable_sha256(spec), oracle=adopted, old_candidate_count=len(old),
        adopted_candidate_count=len(pool), new_candidates_generated_this_campaign=0,
        source_search_queries=freeze["new_graph_oracle_queries"],
        source_generation_oracle="GINE", new_verification_oracle="GIN", pool_sha256=freeze["candidate_universe_sha256"],
        post_hoc_development=True, previous_test_seen=True, test_used_to_design_search=False,
        selector_contract=dict(policy=POLICY, reach_weight=.25, grid_weight=.75,
            threshold_weights="ORIGINAL_GRID_NORMALIZED_TO_ONE", swap_passes=2,
            K10_floor="V1_LEGAL_S10_CALIBRATION_THETA_COVERAGE", S10_subset_S20=True),
        old_order=old_freeze["ordered_rule_ids"], old_selected_variant=old_freeze["selector_details"]["selected_variant"],
        search_addition_state="NOT_STARTED_WAITING_COMPLETE_TRAIN_CEILING", main_matrix_write=False))


def admission(spec, stage):
    out = Path(spec["output_root"])
    stat = os.statvfs(out)
    free = stat.f_bavail * stat.f_frsize
    budget = int(spec["stage_persistent_peak_bytes"][stage])
    reserve = max(2 * 1024**3, int(free * .2))
    if free < reserve + budget:
        raise RuntimeError(f"STAGE_STORAGE_ADMISSION:{stage}:free={free}:peak={budget}:reserve={reserve}")
    return dict(stage=stage, free_bytes=free, peak_new_bytes=budget, reserve_bytes=reserve,
                inode_source="SENTINEL_NOT_QUOTA" if stat.f_favail > 10**12 else "statvfs",
                sampled_at=utc_now())


def _parent_saved(path, binding):
    item = verified(path)
    if item["binding"] != binding:
        raise ValueError("PARENT_RESUME_BINDING_CHANGED")
    return item


def _old_calibration(spec, index, parent, split="calibration"):
    old_spec = bound_json(spec["v1_spec"])
    root = Path(old_spec["output_root"]) / "ours" / split / f"parent-{index:05d}"
    receipt = read_json(root / "complete.json")
    pairs = root / "pairs.jsonl"
    if (receipt["spec_sha256"] != stable_sha256(old_spec) or receipt["parent_id"] != parent.parent_id
            or receipt["pairs_sha256"] != sha256_file(pairs) or receipt["state"] != "COMPLETE"
            or old_spec["gin_files"] != spec["gin_files"]):
        raise ValueError("V1_CALIBRATION_ADOPTION_BINDING_CONFLICT")
    return read_jsonl(pairs), read_jsonl(root / "applications.jsonl")


def _iter_units(spec, group, split):
    path = Path(spec["output_root"]) / group / split
    for i in range(spec["base_counts"][split]):
        record = verified(path / f"parent-{i:05d}.json")
        if record["spec_sha256"] != stable_sha256(spec):
            raise ValueError("PARENT_UNIT_SPEC_DRIFT")
        yield record


def evaluate(spec, group, split, *, limit=None):
    """Each new-parent call uses own GIN flips and compact immutable JSON."""
    validate(spec)
    contract = plan(spec)
    root = Path(spec["output_root"])
    old, expanded, _ = pools(spec)
    frozen = verified(root / "selection_freeze.json") if split == "test" else None
    if split == "test" and (frozen["spec_sha256"] != stable_sha256(spec) or frozen["test_loaded"] is not False):
        raise ValueError("TEST_BEFORE_GLOBAL_FREEZE")
    if group not in ("old66", "adopted2607") or split not in ("train", "calibration", "test"):
        raise ValueError("INVALID_STAGE")
    candidates = old if group == "old66" else expanded
    if frozen:
        wanted = set().union(*(set(v) for v in frozen["controls"].values()))
        candidates = [c for c in expanded if c["candidate_id"] in wanted]
    runtime_spec = copy.deepcopy(spec)
    fp = root / "selection_freeze.json"
    if split == "test":
        from src.ablations.gnn.reach_raw_distance_reuse import build_index
        index = root / "raw_test_adoption.json"
        build_index(spec["raw_distance_source"], split="test", output=index,
            repo=Path(__file__).resolve().parents[2], test_freeze_path=fp,
            test_freeze_sha=sha256_file(fp), validate_test_freeze=lambda f: require_freeze(spec, f))
        runtime_spec["raw_cost_indexes"]["test"] = dict(path=str(index), sha256=sha256_file(index),
            new_test_freeze_path=str(fp), new_test_freeze_sha256=sha256_file(fp))
    parents = adapter.fixed_source_parents(spec, split, test_authorized=frozen is not None)
    target = root / group / split
    target.mkdir(parents=True, exist_ok=True)
    stage = f"{group}-{split}"
    if (target / "terminal.json").exists():
        return verified(target / "terminal.json")
    resource_receipt = admission(spec, stage)
    runtime = adapter.build_runtime(runtime_spec, target / "runtime", split=split,
        test_freeze=frozen, validate_test_freeze=lambda f: require_freeze(spec, f))
    from src.eval.bace_reach_v2 import evaluate_pairs
    old_ids = {r["candidate_id"] for r in old}
    started = time.monotonic()
    try:
        with (target / "writer.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for i, parent in enumerate(parents[:limit] if limit else parents):
                binding = stable_sha256(dict(spec=stable_sha256(spec), group=group, split=split,
                    parent_id=parent.parent_id, pool=[r["candidate_id"] for r in candidates], freeze=frozen))
                path = target / f"parent-{i:05d}.json"
                if path.exists():
                    _parent_saved(path, binding)
                    continue
                seed_pairs, seed_matches = [], []
                todo = candidates
                if split == "calibration":
                    seed_pairs, seed_matches = _old_calibration(spec, i, parent)
                    todo = [r for r in candidates if r["candidate_id"] not in old_ids]
                elif split == "test":
                    # This code is reached only after this campaign's freeze.
                    # Same GIN and same original operations: reuse sealed V1
                    # selected-parent units, not old GINE masks or minima.
                    seed_pairs, seed_matches = _old_calibration(spec, i, parent, "test")
                    reused = {r["candidate_id"] for r in seed_pairs}
                    if not reused <= {r["candidate_id"] for r in candidates}:
                        raise ValueError("OLD_CONTROL_TEST_UNION_NOT_RETAINED")
                    todo = [r for r in candidates if r["candidate_id"] not in reused]
                elif group == "adopted2607" and split == "train":
                    saved = verified(root / "old66/train" / f"parent-{i:05d}.json")
                    if saved["spec_sha256"] != stable_sha256(spec) or saved["parent_id"] != parent.parent_id:
                        raise ValueError("OLD66_TRAIN_ADOPTION_BINDING")
                    seed_pairs, seed_matches = saved["pair_rows"], saved["match_rows"]
                    todo = [r for r in candidates if r["candidate_id"] not in old_ids]
                pairs, matches = ([], [])
                if todo:
                    pairs, matches = evaluate_pairs([parent], todo, oracle=runtime["oracle"],
                        featurizer=runtime["featurizer"], distance_provider=runtime["distance"], split=split,
                        oracle_checkpoint_id=runtime["oracle"].checkpoint_id,
                        oracle_batch_size=spec["batch_size"])
                    for row in (*pairs, *matches):
                        row.update(split=split, oracle_backbone=runtime["oracle"].backbone,
                                   oracle_temperature=runtime["oracle"].temperature)
                all_pairs = seed_pairs + pairs
                expected = {r["candidate_id"] for r in candidates}
                if len(all_pairs) != len(expected) or {r["candidate_id"] for r in all_pairs} != expected:
                    raise ValueError("FULL_POOL_CARTESIAN_COVERAGE_FAILED")
                seal(path, dict(binding=binding, spec_sha256=stable_sha256(spec), parent_id=parent.parent_id,
                    group=group, split=split, pair_rows=all_pairs, match_rows=seed_matches + matches,
                    old_gine_flip_masks_reused=False, reused_same_gin_pairs=len(seed_pairs)))
                atomic_json(target / "progress.json", dict(state="RUNNING", completed_parents=i+1,
                    total_parents=len(parents), science_pid=os.getpid(), updated_at=utc_now(),
                    elapsed_seconds=time.monotonic()-started))
                if (i + 1) % 16 == 0:
                    admission(spec, stage)
        result = dict(state="BOUNDED_PROBE_COMPLETE" if limit else "PARENT_EVALUATION_COMPLETE",
            spec_sha256=stable_sha256(spec), parent_count=min(limit or len(parents), len(parents)),
            candidate_count=len(candidates), elapsed_seconds=time.monotonic()-started,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            raw_distance_stats=runtime["distance"].stats_dict(), resource_admission=resource_receipt,
            test_loaded=split=="test", main_matrix_write=False)
        return seal(target / ("probe.json" if limit else "terminal.json"), result)
    finally:
        runtime["distance"].close()


def ceiling(spec, group, split):
    contract = plan(spec)
    all_rows = []
    ids = []
    native = 0
    for record in _iter_units(spec, group, split):
        ids.append(record["parent_id"])
        all_rows.extend(record["pair_rows"])
        native += int(record["pair_rows"][0]["pred_before"]) == 1
    from src.eval.bace_reach_v2 import summary
    thresholds = bound_json(spec["thresholds"])
    result = dict(group=group, split=split, parent_count=len(ids), native_source_count=native,
        counts=summary(all_rows, ids, contract["old_order"], thresholds),
        old_gine_flips_used=False, test_loaded=False)
    seal(Path(spec["output_root"]) / f"ceiling-{group}-{split}.json", result)
    return result


def freeze(spec):
    root = Path(spec["output_root"])
    contract = plan(spec)
    if (root / "selection_freeze.json").exists():
        value = verified(root / "selection_freeze.json")
        require_freeze(spec, value)
        return value
    gate = verified(root / "train_adoption_gate.json")
    if gate["spec_sha256"] != stable_sha256(spec) or gate["state"] != "NO_TRAIN_REACH_GAP_REQUIRING_SUPPLEMENT":
        raise ValueError("TRAIN_ONLY_SUPPLEMENT_MUST_CLOSE_BEFORE_FINAL_SELECTOR")
    old, pool, _ = pools(spec)
    from src.eval.bace_reach_selector import ReachMasks
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    pairs, ids = [], []
    for record in _iter_units(spec, "adopted2607", "calibration"):
        pairs.extend(record["pair_rows"])
        ids.append(record["parent_id"])
    matrix = matrix_from_pairs(ids, pool, pairs, root=root, split="calibration")
    thresholds = bound_json(spec["thresholds"])
    new = select(ReachMasks.from_distances([r["candidate_id"] for r in pool], matrix.distances, thresholds), contract["old_order"])
    oldset = {r["candidate_id"] for r in old}
    oldmatrix = matrix_from_pairs(ids, old, [r for r in pairs if r["candidate_id"] in oldset], root=root, split="calibration")
    control = select(ReachMasks.from_distances([r["candidate_id"] for r in old], oldmatrix.distances, thresholds), contract["old_order"])
    return seal(root / "selection_freeze.json", dict(state="CALIBRATION_SELECTOR_FROZEN", spec_sha256=stable_sha256(spec),
        contract_sha256=contract["self_sha256"], policy=POLICY, test_loaded=False, main_matrix_write=False,
        controls=dict(old66_old_selector=contract["old_order"], old66_new_selector=control["ordered_rule_ids"],
            adopted2607_new_selector=new["ordered_rule_ids"]), selection_details=dict(old=control, new=new),
        calibration_parent_ids=ids, created_at=utc_now()))


def require_freeze(spec, frozen):
    if (frozen.get("state") != "CALIBRATION_SELECTOR_FROZEN" or frozen.get("test_loaded") is not False
            or frozen.get("spec_sha256") != stable_sha256(spec) or frozen.get("policy") != POLICY
            or set(frozen.get("controls", {})) != {"old66_old_selector", "old66_new_selector", "adopted2607_new_selector"}):
        raise ValueError("A_PLUS_ACTUAL_GLOBAL_FREEZE_REQUIRED")


def aggregate(spec):
    root = Path(spec["output_root"])
    frozen = verified(root / "selection_freeze.json")
    require_freeze(spec, frozen)
    records = list(_iter_units(spec, "adopted2607", "test"))
    parents = [r["parent_id"] for r in records]
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    th = threshold_bundle_from_dict(bound_json(spec["thresholds"]))
    out = {}
    for name, order in frozen["controls"].items():
        pairs = [p for r in records for p in r["pair_rows"] if p["candidate_id"] in order]
        out[name] = prefix_metrics(parents, order, pairs, theta=th.theta_star, cap=th.cost_cap, endpoints=th.raw_thresholds)
    seal(root / "metrics.json", dict(state="EVALUATED_NOT_INDEPENDENT_AUDIT", results=out,
        spec_sha256=stable_sha256(spec), freeze_sha256=frozen["self_sha256"], main_matrix_write=False))
    rows = [dict(variant=v, **r) for v, result in out.items() for r in result["prefix_rows"]]
    atomic_csv(root / "source_csv/ours_variant_comparison.csv", rows)
    for filename, key in (("figure3_k1_20.csv", "prefix_rows"), ("figure4_exact_ecdf.csv", "exact_ecdf"),
                          ("parent_best_distances.csv", "parent_distances")):
        rows = out["adopted2607_new_selector"][key]
        atomic_csv(root / "source_csv" / filename, rows)
    return {name: [r for r in result["prefix_rows"] if r["cohort"]=="fixed141" and r["K_requested"] in (10,20)] for name,result in out.items()}


def status(spec):
    root = Path(spec["output_root"])
    result = dict(experiment_id=EXPERIMENT, root=str(root), main_matrix_write=False)
    for group, split in (("old66","train"),("old66","calibration"),("adopted2607","train"),
                         ("adopted2607","calibration"),("adopted2607","test")):
        path = root/group/split
        result[f"{group}/{split}"] = read_json(path/"terminal.json") if (path/"terminal.json").exists() else (
            read_json(path/"progress.json") if (path/"progress.json").exists() else "NOT_STARTED")
    result["selector_frozen"]=(root/"selection_freeze.json").exists()
    result["metrics_available"]=(root/"metrics.json").exists()
    return result


def audit(spec):
    """Stream saved parent records; independently reduce masks and final metrics."""
    from src.experiments.bace_gin_audit import audit_parent_rows, recompute_metrics, equal
    root = Path(spec["output_root"])
    frozen = verified(root / "selection_freeze.json")
    require_freeze(spec, frozen)
    adopted = adapter.validate_gin_adoption(spec)
    oracle = dict(model_sha256=adopted["model_sha256"], temperature=adopted["temperature"])
    inventories, results = {}, []
    for split in ("calibration", "test"):
        for unit in _iter_units(spec, "adopted2607", split):
            rows, apps = unit["pair_rows"], unit["match_rows"]
            result = audit_parent_rows(rows, apps, method="ours", split=split,
                parent_id=unit["parent_id"], parent_smiles=rows[0]["parent_smiles"],
                candidate_ids=[r["candidate_id"] for r in rows], oracle=oracle)
            results.append(dict(split=split, parent_id=unit["parent_id"], **result))
    metrics = verified(root / "metrics.json")
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    th = threshold_bundle_from_dict(bound_json(spec["thresholds"]))
    test = list(_iter_units(spec, "adopted2607", "test"))
    for name, order in frozen["controls"].items():
        pairs = [p for u in test for p in u["pair_rows"] if p["candidate_id"] in order]
        independent = recompute_metrics([u["parent_id"] for u in test], order, pairs,
            theta=th.theta_star, cap=th.cost_cap, endpoints=th.raw_thresholds)
        actual = metrics["results"][name]
        for key in ("prefix_rows", "exact_ecdf", "parent_distances"):
            if len(actual[key]) != len(independent[key]):
                raise ValueError("METRIC_ROW_COUNT_CONFLICT")
            for a, b in zip(actual[key], independent[key]):
                for field, value in b.items():
                    equal(a[field], value, f"{name}/{key}/{field}")
    receipt = dict(state="SAVED_RECORD_AND_METRIC_CONSISTENCY_PASS", spec_sha256=stable_sha256(spec),
        freeze_sha256=frozen["self_sha256"], metrics_sha256=metrics["self_sha256"],
        parent_units=results, model_inference_rerun=False, ot_recomputed=0,
        audit_scope="SAVED_APPLICATIONS_AND_INDEPENDENT_METRIC_REDUCER_NOT_MODEL_REEXECUTION",
        main_matrix_write=False, new_method_final_science_claim="POST_HOC_FROZEN_GIN_REACH_AWARE")
    return seal(root / "audit/final_audit.json", receipt)


def train_gate(spec):
    """A train-only decision, never derived from the new calibration outcome."""
    root = Path(spec["output_root"])
    contract = plan(spec)
    terminal = verified(root / "adopted2607/train/terminal.json")
    if terminal["parent_count"] != 386 or terminal["spec_sha256"] != stable_sha256(spec):
        raise ValueError("FULL_386_TRAIN_EVIDENCE_REQUIRED")
    eligible, reached, uncovered = [], [], []
    for unit in _iter_units(spec, "adopted2607", "train"):
        rows = unit["pair_rows"]
        if len(rows) != contract["adopted_candidate_count"] or len({r["pred_before"] for r in rows}) != 1:
            raise ValueError("TRAIN_FULL_POOL_OR_SOURCE_PREDICTION_CONFLICT")
        if rows[0]["pred_before"] != 1:
            continue
        eligible.append(unit["parent_id"])
        (reached if any(r["pair_strict_flip"] for r in rows) else uncovered).append(unit["parent_id"])
    return seal(root / "train_adoption_gate.json", dict(
        state="SUPPLEMENTAL_TRAIN_SEARCH_REQUIRED" if uncovered else "NO_TRAIN_REACH_GAP_REQUIRING_SUPPLEMENT",
        spec_sha256=stable_sha256(spec), train_terminal_sha256=terminal["self_sha256"],
        eligible_count=len(eligible), reached_count=len(reached), uncovered_parent_ids=uncovered,
        criterion="ANY_GIN_SOURCE_TRAIN_PARENT_WITHOUT_STRICT_FLIP_IN_COMPLETE_SAVED2607",
        native_normalization_for_diagnosis_only=True, primary_denominator_unchanged=386,
        calibration_used=False, test_used=False, additional_ppo_updates=0))


def audit_calibration(spec):
    """Validate actual new application records before selecting or opening test."""
    from src.experiments.bace_gin_audit import audit_parent_rows
    adopted = adapter.validate_gin_adoption(spec)
    oracle = dict(model_sha256=adopted["model_sha256"], temperature=adopted["temperature"])
    counts = []
    for unit in _iter_units(spec, "adopted2607", "calibration"):
        rows = unit["pair_rows"]
        counts.append(dict(parent_id=unit["parent_id"], **audit_parent_rows(rows, unit["match_rows"],
            method="ours", split="calibration", parent_id=unit["parent_id"],
            parent_smiles=rows[0]["parent_smiles"], candidate_ids=[r["candidate_id"] for r in rows], oracle=oracle)))
    return seal(Path(spec["output_root"]) / "audit/calibration_records.json", dict(
        state="SAVED_CALIBRATION_RECORD_CONSISTENCY_PASS", spec_sha256=stable_sha256(spec),
        parent_units=counts, model_inference_rerun=False, ot_recomputed=0, test_loaded=False))
