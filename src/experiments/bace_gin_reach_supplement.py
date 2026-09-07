"""Bounded, train-only GIN search after complete frozen2607 verification.

This leaf does not load a selector, MolCLR, OT, calibration, or test payload.
The original source experiment and its2607 pool remain immutable. A subsequent
driver must explicitly adopt this new pool before any new calibration stage.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from src.chem.bace_reach_search import SearchBudget, new_graph_key, retain_train_pool, search_parent
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file, stable_sha256, utc_now
from src.experiments import bace_gin_ours as adapter
from src.experiments.bace_gin_reach_v2 import pools, seal, validate, verified

METHOD = "BACE_GIN_TRAIN_ONLY_REACH_SUPPLEMENT_V1"


def prediction(probabilities, label):
    p = [float(x) for x in probabilities]
    if (len(p) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
            or not math.isclose(sum(p), 1., abs_tol=1e-5, rel_tol=0)
            or int(label) != (0 if p[0] >= p[1] else 1)):
        raise ValueError("INVALID_BOUND_GIN_PROBABILITY")
    return dict(predicted_label=int(label), probabilities=p)


def oracle_identity(adoption):
    return stable_sha256({k: adoption[k] for k in (
        "model_sha256", "temperature_sha256", "temperature", "feature_schema_sha256")})


def inspect_parent(unit, *, spec_sha, parent, candidate_ids, adoption):
    """Validate one complete parent unit; reuse only its own GIN probabilities."""
    if (unit.get("spec_sha256") != spec_sha or unit.get("parent_id") != parent.parent_id
            or unit.get("split") != "train" or unit.get("group") != "adopted2607"
            or unit.get("old_gine_flip_masks_reused") is not False):
        raise ValueError("SOURCE_UNIT_SCOPE_CONFLICT")
    pairs, matches = unit["pair_rows"], unit["match_rows"]
    if len(pairs) != len(candidate_ids) or Counter(r["candidate_id"] for r in pairs) != Counter(candidate_ids):
        raise ValueError("SOURCE_FULL2607_COVERAGE_INCOMPLETE")
    expected_binding = stable_sha256(dict(spec=spec_sha, group="adopted2607", split="train",
        parent_id=parent.parent_id, pool=list(candidate_ids), freeze=None))
    if unit.get("binding") != expected_binding:
        raise ValueError("SOURCE_UNIT_INPUT_BINDING_CONFLICT")
    candidate_set = set(candidate_ids)
    counts, flip_counts = Counter(), Counter()
    cache, conflicts = {}, set()
    p1 = float(pairs[0]["p1_before"])
    before = prediction([1. - p1, p1], pairs[0]["pred_before"])
    for row in (*pairs, *matches):
        if (row.get("parent_id") != parent.parent_id or row.get("parent_smiles") != parent.smiles
                or row.get("candidate_id") not in candidate_set
                or row.get("oracle_checkpoint_hash") != adoption["model_sha256"]
                or row.get("oracle_backbone") != "gin"
                or row.get("oracle_temperature") != adoption["temperature"]
                or row.get("split") != "train" or row.get("rf_oracle_used") is not False
                or int(row["pred_before"]) != before["predicted_label"]
                or float(row["p1_before"]) != p1):
            raise ValueError("SOURCE_GIN_GRAPH_OR_PROBABILITY_BINDING_CONFLICT")
    for row in matches:
        counts[row["candidate_id"]] += 1
        if row.get("p_before") is not None:
            actual_before = prediction(row["p_before"], row["pred_before"])
            if actual_before["probabilities"][1] != p1:
                raise ValueError("PARENT_PROBABILITY_CONFLICT")
            before = actual_before
        if not row["delete_valid"]:
            if row["teacher_strict_flip"]:
                raise ValueError("INVALID_DELETION_MARKED_FLIP")
            continue
        if (not row.get("residual_smiles") or row.get("residual_connected") is not True
                or row.get("sanitize_ok") is not True or row.get("residual_num_components") != 1
                or row.get("contains_dot") is not False or row.get("p_after") is None):
            raise ValueError("VALID_RESIDUAL_PREDICTION_MISSING")
        after = prediction(row["p_after"], row["pred_after"])
        flip = before["predicted_label"] == 1 and after["predicted_label"] == 0
        if bool(row["teacher_strict_flip"]) != flip:
            raise ValueError("SOURCE_STRICT_FLIP_CONFLICT")
        flip_counts[row["candidate_id"]] += int(flip)
        key = new_graph_key(row["residual_smiles"], oracle_identity(adoption))
        if key in cache and cache[key] != after:
            # Do not average or select saved probabilities. A later fresh query
            # is counted if a graph has genuinely conflicting saved predictions.
            conflicts.add(key)
        cache[key] = after
    for row in pairs:
        if (row["num_matches"] != counts[row["candidate_id"]]
                or row["num_strict_flip_matches"] != flip_counts[row["candidate_id"]]):
            raise ValueError("SOURCE_MATCH_COVERAGE_OR_FLIP_COUNT_CONFLICT")
    for key in conflicts:
        cache.pop(key, None)
    return dict(before=before, covered=bool(sum(flip_counts.values())), cache=cache,
        conflicting_saved_graphs=len(conflicts), match_count=len(matches))


def seeded_extra_ids(states: Sequence[Mapping[str, Any]], limit=128):
    if limit != 128 or len({s["parent_id"] for s in states}) != len(states):
        raise ValueError("EXTRA_PARENT_BOUND_OR_ID_CONFLICT")
    ids = sorted(s["parent_id"] for s in states if not s["witnesses"])
    rng = random.Random(7)  # local RNG; no mutation of model/global RNG state
    rng.shuffle(ids)
    return ids[:limit]


def load_inputs(spec, output):
    validate(spec)
    source = Path(spec["output_root"]).resolve()
    output = Path(output).resolve()
    allowed = "/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments/bace-gin-reach-aligned-v2/"
    if (not str(output).startswith(allowed) or output == source
            or output in source.parents or output == source / "adopted2607"):
        raise ValueError("SUPPLEMENT_OUTPUT_MUST_BE_FRESH_SCOPED_ROOT")
    if output.exists() and any(output.iterdir()) and not (output / "supplement_contract.json").is_file():
        raise ValueError("SUPPLEMENT_OUTPUT_ALREADY_OCCUPIED")
    old, pool, source_freeze = pools(spec)
    if len(old) != 66 or len(pool) != 2607 or len({r["candidate_id"] for r in pool}) != 2607:
        raise ValueError("EXACT_EXISTING2607_POOL_REQUIRED")
    source_contract = verified(source / "contract.json")
    if source_contract["spec_sha256"] != stable_sha256(spec):
        raise ValueError("SOURCE_EXPERIMENT_CONTRACT_CONFLICT")
    search = verified(spec["reach_search_contract"])
    budget = SearchBudget(**search["search_budget"])
    # Copy the historical size contract, rather than selecting a broader new one.
    if not search.get("size_contract_source") or search.get("test_opened") is not False:
        raise ValueError("SOURCE_SIZE_OR_TRAIN_CONTRACT_MISSING")
    terminal_path = source / "adopted2607/train/terminal.json"
    terminal = verified(terminal_path)
    if (terminal.get("state") != "PARENT_EVALUATION_COMPLETE"
            or terminal.get("spec_sha256") != stable_sha256(spec)
            or terminal.get("parent_count") != 386 or terminal.get("candidate_count") != 2607
            or terminal.get("test_loaded") is not False):
        raise ValueError("COMPLETE_GIN_TRAIN2607_REQUIRED_BEFORE_SEARCH")
    parents = adapter.fixed_source_parents(spec, "train")
    adoption = adapter.validate_gin_adoption(spec)
    return dict(source=source, output=output, pool=pool, source_freeze=source_freeze,
        source_contract=source_contract, search=search, budget=budget, terminal=terminal,
        terminal_path=terminal_path, parents=parents, adoption=adoption)


def _inspect(inputs, spec, index):
    parent = inputs["parents"][index]
    path = inputs["source"] / "adopted2607/train" / f"parent-{index:05d}.json"
    unit = verified(path)
    info = inspect_parent(unit, spec_sha=stable_sha256(spec), parent=parent,
        candidate_ids=[r["candidate_id"] for r in inputs["pool"]], adoption=inputs["adoption"])
    return path, unit, info


def prepare(spec, output):
    inputs = load_inputs(spec, output)
    target = inputs["output"]
    if (target / "supplement_contract.json").exists():
        existing = verified(target / "supplement_contract.json")
        if (existing["source_spec_sha256"] != stable_sha256(spec)
                or existing["source_pool_sha256"] != inputs["source_freeze"]["candidate_universe_sha256"]
                or existing["source_train_terminal_sha256"] != inputs["terminal"]["self_sha256"]
                or existing["search_budget"] != asdict(inputs["budget"])
                or existing["adoption"] != inputs["adoption"]):
            raise ValueError("SUPPLEMENT_SPEC_CHANGED")
        return existing, inputs
    eligible, bindings = [], []
    counts = dict(base_train_parents=386, gin_source_parents=0, already_reachable=0,
                  search_eligible=0, saved_prediction_conflicts=0)
    for index, parent in enumerate(inputs["parents"]):
        path, unit, info = _inspect(inputs, spec, index)
        source_eligible = info["before"]["predicted_label"] == 1
        counts["gin_source_parents"] += int(source_eligible)
        counts["already_reachable"] += int(info["covered"])
        counts["saved_prediction_conflicts"] += info["conflicting_saved_graphs"]
        bindings.append(dict(index=index, parent_id=parent.parent_id, unit_sha256=unit["self_sha256"]))
        if source_eligible and not info["covered"]:
            eligible.append(dict(index=index, parent_id=parent.parent_id, unit_sha256=unit["self_sha256"],
                initial_cache_entries=len(info["cache"])))
    counts["search_eligible"] = len(eligible)
    value = dict(method=METHOD, state="READY" if eligible else "NO_TRAIN_REACH_GAP",
        source_spec_sha256=stable_sha256(spec), source_contract_sha256=inputs["source_contract"]["self_sha256"],
        source_pool_sha256=inputs["source_freeze"]["candidate_universe_sha256"], source_candidate_count=2607,
        source_train_terminal_sha256=inputs["terminal"]["self_sha256"], train_unit_bindings=bindings,
        source_train_parent_ids=[p.parent_id for p in inputs["parents"]], eligible=eligible, counts=counts,
        adoption=inputs["adoption"], oracle_binding=oracle_identity(inputs["adoption"]), source_label=1,
        search_budget=asdict(inputs["budget"]), size_contract_source=inputs["search"]["size_contract_source"],
        seed=7, extra_order="SORT_PARENT_ID_THEN_LOCAL_PYTHON_RANDOM_7_SHUFFLE",
        source_generation_oracle="GINE", supplement_oracle="GIN",
        previous_generation_queries=inputs["source_freeze"]["new_graph_oracle_queries"],
        previous_generation_queries_separately_disclosed=True, previous_gine_search_states_reused=False,
        calibrated_gin_probabilities_reused_only_under_same_model_temperature_schema=True,
        additional_ppo_updates=0, calibration_loaded=False, test_loaded=False,
        main_matrix_write=False, created_at=utc_now())
    target.mkdir(parents=True, exist_ok=True)
    return seal(target / "supplement_contract.json", value), inputs


def load_predictor(spec, adoption):
    """Only a frozen GIN and its featurizer; no distance or selector factory."""
    from src.ablations.gnn.cpu_evaluation import _featurizer
    from src.oracles.gnn_oracle import GNNOracle
    from src.eval.bace_reach_v2 import predict_smiles
    root, manifest = adapter.original_bundle(spec)
    if manifest["files"][manifest["feature_schema_path"]]["sha256"] != adoption["feature_schema_sha256"]:
        raise ValueError("GIN_FEATURE_SCHEMA_DRIFT")
    oracle = GNNOracle.from_checkpoint(spec["gin_root"], device="cpu", verify_hashes=False,
        batch_size=int(spec.get("batch_size", 256)))
    if (oracle.backbone != "gin" or oracle.checkpoint_id != adoption["model_sha256"]
            or oracle.temperature != adoption["temperature"] or oracle.source_label != 1 or oracle.num_classes != 2):
        raise ValueError("ACTUAL_GIN_DIFFERS_FROM_SUPPLEMENT_BINDING")
    for parameter in oracle.model.parameters():
        parameter.requires_grad_(False)
    oracle.model.eval()
    features = _featurizer(root, manifest)
    return lambda smiles: predict_smiles(oracle, features, smiles, "train_gin_supplement")


def _pass(target, contract, parent, info, *, predict, pool, maximum, previous=None):
    number = 1 if previous is None else 2
    if maximum != (128 if number == 1 else 384) or (previous is not None
            and not 0 <= int(previous["new_graph_oracle_queries"]) <= 128):
        raise ValueError("SUPPLEMENT_PARENT_PASS_SEQUENCE_OR_QUERY_BUDGET_EXCEEDED")
    path = target / f"parent-{stable_sha256(parent.parent_id)[:24]}-pass{number}.json"
    binding = dict(contract_sha256=contract["self_sha256"], parent_id=parent.parent_id,
        source_unit_sha256=info["source_unit_sha256"], previous_sha256=stable_sha256(previous) if previous else None,
        pass_number=number, maximum_new_queries=maximum)
    if path.exists():
        saved = verified(path)
        if saved["binding"] != binding:
            raise ValueError("SUPPLEMENT_PARENT_PASS_BINDING_CHANGED")
        return saved["search"]
    intent = path.with_suffix(".intent.json")
    if intent.exists():
        raise ValueError("UNCOMMITTED_QUERY_PASS_REQUIRES_EXPLICIT_ACCOUNTING:" + str(intent))
    seal(intent, dict(binding=binding, state="PASS_STARTED", started_at=utc_now(), science_pid=os.getpid()))
    state = search_parent(parent_id=parent.parent_id, parent_smiles=parent.smiles,
        before=info["before"], predict=predict, old_candidates=pool,
        oracle_binding=contract["oracle_binding"], budget=SearchBudget(**contract["search_budget"]),
        maximum_new_queries=maximum, previous=previous, source_label=1,
        initial_oracle_cache=info["cache"] if previous is None else None)
    bound = 128 if number == 1 else 512
    if state["pass_new_queries"] > maximum or state["new_graph_oracle_queries"] > bound:
        raise ValueError("SUPPLEMENT_ACTUAL_QUERY_BUDGET_EXCEEDED")
    seal(path, dict(binding=binding, search=state, state="PASS_COMMITTED", created_at=utc_now()))
    return state


def run(spec, output):
    contract, inputs = prepare(spec, output)
    target = inputs["output"]
    final = target / "candidate_freeze.json"
    if final.exists():
        value = verified(final)
        if value["supplement_contract_sha256"] != contract["self_sha256"]:
            raise ValueError("SUPPLEMENT_FINAL_BINDING_CHANGED")
        return value
    with (target / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not contract["eligible"]:
            return seal(final, dict(state="NO_SUPPLEMENT_REQUIRED", method=METHOD,
                supplement_contract_sha256=contract["self_sha256"], source_spec_sha256=stable_sha256(spec),
                source_candidate_count=2607, candidate_count=2607, new_candidate_count=0,
                candidate_universe_path=str(Path(spec["reach_candidate_file"]).resolve()),
                candidate_universe_sha256=inputs["source_freeze"]["candidate_universe_sha256"],
                old2607_content_unchanged=True, new_graph_oracle_queries=0,
                previous_generation_queries=contract["previous_generation_queries"],
                extra_parent_ids=[], parent_query_counts={}, actual_query_budget_exceeded=False,
                calibration_loaded=False, test_loaded=False, additional_ppo_updates=0,
                main_matrix_write=False, created_at=utc_now()))
        predict = load_predictor(spec, inputs["adoption"]) if contract["eligible"] else None
        states, initial_cache_count, initial_queries, initial_hits = [], 0, {}, {}
        for entry in contract["eligible"]:
            _, unit, info = _inspect(inputs, spec, entry["index"])
            if unit["self_sha256"] != entry["unit_sha256"]:
                raise ValueError("SOURCE_TRAIN_UNIT_CHANGED_AFTER_PLAN")
            info["source_unit_sha256"] = unit["self_sha256"]
            parent = inputs["parents"][entry["index"]]
            states.append(_pass(target, contract, parent, info, predict=predict, pool=inputs["pool"], maximum=128))
            initial_queries[parent.parent_id] = states[-1]["pass_new_queries"]
            initial_hits[parent.parent_id] = states[-1]["pass_cache_hits"]
            initial_cache_count += len(info["cache"])
            atomic_json(target / "progress.json", dict(state="INITIAL_SEARCH", completed_parents=len(states),
                total_parents=len(contract["eligible"]), new_graph_queries=sum(s["new_graph_oracle_queries"] for s in states),
                science_pid=os.getpid(), updated_at=utc_now()))
        selected = seeded_extra_ids(states)
        seal(target / "extra_parent_order.json", dict(contract_sha256=contract["self_sha256"],
            parent_ids=selected, seed=7, first_pass_sha256=stable_sha256(states), calibration_loaded=False, test_loaded=False))
        position = {s["parent_id"]: index for index, s in enumerate(states)}
        entries = {r["parent_id"]: r for r in contract["eligible"]}
        for parent_id in selected:
            entry = entries[parent_id]
            _, unit, info = _inspect(inputs, spec, entry["index"])
            if unit["self_sha256"] != entry["unit_sha256"]:
                raise ValueError("SOURCE_TRAIN_UNIT_CHANGED_AFTER_PLAN")
            info["source_unit_sha256"] = unit["self_sha256"]
            index = position[parent_id]
            states[index] = _pass(target, contract, inputs["parents"][entry["index"]], info,
                predict=predict, pool=inputs["pool"], maximum=384, previous=states[index])
        pool = retain_train_pool(inputs["pool"], states, 4096)
        if pool[:2607] != inputs["pool"] or len({p["candidate_id"] for p in pool}) != len(pool):
            raise ValueError("ORIGINAL2607_OR_UNIQUE_CANDIDATE_ID_CHANGED")
        atomic_jsonl(target / "candidate_universe.jsonl", pool)
        return seal(final, dict(state="TRAIN_ONLY_POOL_FROZEN", method=METHOD,
            supplement_contract_sha256=contract["self_sha256"], source_spec_sha256=stable_sha256(spec),
            source_candidate_count=2607, candidate_count=len(pool), new_candidate_count=len(pool)-2607,
            candidate_universe_path=str(target / "candidate_universe.jsonl"),
            candidate_universe_sha256=sha256_file(target / "candidate_universe.jsonl"),
            initial_parent_count=len(states), extra_parent_ids=selected, seed=7,
            new_graph_oracle_queries=sum(s["new_graph_oracle_queries"] for s in states),
            reused_initial_prediction_entries=initial_cache_count,
            parent_query_counts={s["parent_id"]: s["new_graph_oracle_queries"] for s in states},
            first_pass_query_counts=initial_queries,
            extra_pass_query_counts={s["parent_id"]: s["pass_new_queries"] for s in states if s["parent_id"] in selected},
            cache_hits=sum(initial_hits.values()) + sum(s["pass_cache_hits"] for s in states if s["parent_id"] in selected),
            parents_with_witness=sum(bool(s["witnesses"]) for s in states),
            actual_query_budget_exceeded=False, previous_generation_queries=contract["previous_generation_queries"],
            old2607_content_unchanged=True, calibration_loaded=False, test_loaded=False,
            additional_ppo_updates=0, main_matrix_write=False,
            class_level_calibration_coverage_not_yet_evaluated=True, created_at=utc_now()))


def status(output):
    root = Path(output)
    for name in ("candidate_freeze.json", "progress.json", "supplement_contract.json"):
        if (root / name).is_file():
            return read_json(root / name)
    return dict(state="NOT_STARTED")
