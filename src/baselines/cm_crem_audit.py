"""BACE-specific, saved-record provenance audit for the independent CM route.

Saved-record review executes no science. The separate independent spotcheck
rebuilds a fixed sample with the original frozen models and exact OT. Hash-closed
producer records are mandatory: a raw pair key excludes its result value and is
not by itself proof that a distance was computed. Neither is portable PASS.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import hashlib
import math
import os
from pathlib import Path
import re
import socket
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from src.baselines.cm_crem_runtime import digest, file_sha
from src.baselines.cm_crem_selection import PrefixEvaluation, SelectionFreeze, evaluate_frozen_test, select_calibration

METHOD = "CM-CReM-Global-Budgeted-v1"
UPSTREAM = "b5816b502cde00ee24c652a02cbc54664583f773"
SOURCE_SHA = "b5e485195ede009c560ee397f458794dc22272c14fb17faf3313fbdc64f39e49"
GENERATION_TERMINALS = {"GENERATED", "NO_NATIVE_REPLACEMENT", "NO_REPLACEABLE_CONTEXT", "TIMEOUT_BUDGETED"}


class ProvenanceAuditError(ValueError):
    """Missing/contradictory evidence, never a zero result or scientific PASS."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceAuditError(message)


def _sha(value: Any, role: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"Missing SHA-256: {role}")
    return value


def _path(root: Path, relative: str) -> Path:
    part = Path(relative)
    _require(not part.is_absolute() and ".." not in part.parts and bool(part.parts), "Record path escapes run")
    result = root / part
    _require(not any(p.is_symlink() for p in (result, *result.parents)), "Symlinked audit evidence")
    return result


def _record_stages(name: str) -> set[str]:
    exact = {"pilot/oracle.json": {"pilot-oracle"}, "pilot/final_receipt.json": {"pilot-closeout"},
        "pilot/pool.json": {"pilot-filter"}, "attribution.json": {"attribution"},
        "pool_freeze.json": {"filter"}, "pool_encodings.json": {"encode"},
        "selection_freeze.json": {"select"}}
    if name in exact:
        return exact[name]
    for prefix, stages in (("attribution_units/", {"pilot-oracle", "attribution"}),
                           ("generation_units/", {"pilot-generate", "generate"}),
                           ("filter_units/", {"pilot-filter", "filter"}),
                           ("calibration/", {"calibrate"}), ("test/", {"test"})):
        if name.startswith(prefix):
            return stages
    return set()


def _verify_closed_jobs(jobs: set[str]) -> dict[str, str]:
    """Small read-only scheduler query, not a controller or polling loop."""
    query = subprocess.run(["sacct", "-X", "-j", ",".join(sorted(jobs)), "--noheader", "--parsable2",
                            "--format=JobIDRaw,State,ExitCode"], text=True, capture_output=True,
                           check=True, timeout=30)
    states = {}
    for line in query.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] in jobs:
            state = fields[1].split()[0].rstrip("+")
            # A durable, legal unit from a subsequently failed job is reusable;
            # the unit closure and its scientific terminal are checked below.
            terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
                        "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
            _require(state in terminal and (state != "COMPLETED" or fields[2] == "0:0"),
                     f"Producer job is not terminal: {line}")
            states[fields[0]] = state + ":" + fields[2]
    _require(set(states) == jobs, "Missing real Slurm completion evidence for a scientific producer")
    return states


def validate_generation(record: Mapping[str, Any], attribution: Mapping[str, Any], science_hash: str,
                        *, fixture: bool = False) -> None:
    parent = str(attribution["parent_id"])
    _require(str(record.get("parent_id")) == parent, "Generation parent identity mismatch")
    status = record.get("status")
    _require(status in GENERATION_TERMINALS, f"Generation is not a legal terminal: {status}")
    raw = record.get("retained_raw")
    _require(isinstance(raw, list) and len(raw) <= 128, "Raw parent output exceeds/misses frozen 128 budget")
    if status == "NO_REPLACEABLE_CONTEXT":
        _require(attribution.get("generation_allowed") is False and attribution.get("status") == status
                 and record.get("top_level_calls") == 0 and not raw,
                 "No-context exemption lacks the original no-context attribution")
        return
    _require(attribution.get("generation_allowed") is True, "Generation lacks eligible train attribution")
    if status == "TIMEOUT_BUDGETED":
        _require(not raw and record.get("partial_adopted") is False,
                 "A timed-out parent adopted partial candidates")
        _require(math.isfinite(float(record.get("parent_wall_seconds", -1)))
                 and float(record.get("parent_wall_seconds", -1)) >= 900,
                 "Budgeted timeout lacks the actual 900-second limit")
        _require(int(record.get("worker_pid", 0)) > 0 and record.get("worker_exitcode") is not None,
                 "Timeout has no identified terminal worker")
        return
    _require(record.get("split") == "train" and record.get("science_hash") == science_hash,
             "Generation escaped the train-only science contract")
    seed = int(digest([science_hash, parent, 7])[:16], 16)
    _require(record.get("seed") == seed and record.get("top_level_calls") == 1,
             "Parent seed or one-call budget changed")
    expected_budget = {"radius": 1, "min_max_inc": 3, "max_replacements_per_component": 64,
                       "component_combinations_cap": 500, "retained_raw_limit": 128,
                       "parent_wall_limit_seconds": 900}
    _require(all(record.get("budget", {}).get(k) == v for k, v in expected_budget.items()), "Generation budget drift")
    native = record.get("native", {})
    _require(native.get("upstream_commit") == UPSTREAM and native.get("original_source_sha256") == SOURCE_SHA
             and native.get("native_function_bodies_changed") is False, "Generation is not the pinned official CReM branch")
    expected_env = {"python": "3.11.5", "crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4"}
    _require(record.get("environment") == expected_env, "Generator environment identity missing/changed")
    _require(record.get("worker_exitcode") == 0 and int(record.get("worker_pid", 0)) > 0,
             "Successful generation has no clean real worker exit")
    for query in record.get("query_receipts", []):
        _require(query.get("radius") == 1 and query.get("max_replacements") == 64,
                 "Internal CReM query budget drift")
        _require(query.get("status") in {"COMPLETE", "UPSTREAM_INNER_TIMEOUT"},
                 "Infrastructure/engineering query failure cannot become native empty")
    counts = record.get("counts", {})
    n, unique = counts.get("native_return_count"), counts.get("raw_unique_count")
    _require(type(n) is int and type(unique) is int and n >= unique >= len(raw), "Malformed native/raw funnel counts")
    _require(counts.get("retained_raw_count") == len(raw) == min(128, unique)
             and counts.get("raw_exact_duplicate_count") == n-unique
             and counts.get("raw_truncated_count") == max(0, unique-128), "Raw truncation accounting changed")
    _require((status == "GENERATED") == (n > 0), "Generated versus native-empty status contradicts raw return count")
    _require(len({r.get("smiles") for r in raw}) == len(raw), "Retained raw strings are duplicated")
    _require(raw == sorted(raw, key=lambda r: (digest([seed, r["smiles"]]), r["smiles"])), "Raw retention hash order changed")
    for row in raw:
        _require(row.get("raw_id") == digest([seed, row.get("smiles")]), "Raw identity does not bind parent seed/string")
        _require(type(row.get("source_return_index")) is int and 0 <= row["source_return_index"] < n,
                 "Raw source-return index is invalid")


def validate_encoding(record: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    body = {k: v for k, v in record.items() if k not in {"encoding_sha256", "timing_seconds"}}
    _require(record.get("encoding_sha256") == digest(body), "Encoding content hash mismatch")
    _require(record.get("schema_version") == "cm_crem_original_molclr_node_encoding_v1", "Unknown node encoding schema")
    for key in ("molclr_checkpoint_sha256", "numerical_contract_sha256"):
        _require(record.get(key) == _sha(binding.get(key), key), f"Encoding changed {key}")
    identity = {k: record.get(k) for k in ("canonical_smiles", "graph_sha256", "feature_schema_sha256")}
    _require(record.get("candidate_id") == digest(identity), "Encoding graph identity mismatch")
    array = np.asarray(record.get("H"), dtype=np.float32)
    _require(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0
             and np.isfinite(array).all() and len(record.get("atom_numbers", [])) == array.shape[0],
             "Malformed/nonfinite original node encoding")
    _require(record.get("embedding_dtype") == "float32" and bool(record.get("node_extraction_version")),
             "Node extraction/dtype identity absent")
    producer = record.get("producer", {})
    _require(all(producer.get(k) for k in ("device", "torch", "numpy", "rdkit", "architecture")),
             "Node encoder producer is not bound")


def validate_raw_pair(raw: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any],
                      binding: Mapping[str, Any], *, fixture: bool = False) -> None:
    for row in (left, right):
        validate_encoding(row, binding)
    _require(raw.get("schema_version") == "cm_crem_raw_fullgraph_wnode_v1" and raw.get("solver") == "exact_emd2",
             "Raw distance is not the frozen exact WNode solver")
    _require(raw.get("parent_graph_id") == left["candidate_id"] and raw.get("prototype_graph_id") == right["candidate_id"],
             "Raw distance graph pair differs from stored encodings")
    _require(raw.get("encoding_ids") == sorted([left["encoding_sha256"], right["encoding_sha256"]]),
             "Raw distance refers to an unbound encoding")
    numerical = {key: binding[key] for key in ("numerical_contract_sha256", "feature_cost", "node_mass", "size_penalty_beta")}
    _require(raw.get("numerical_contract") == numerical, "Raw distance numerical contract drift")
    for key in ("producer", "node_extraction_version"):
        _require(left.get(key) == right.get(key), f"Pair encoded under different {key}")
    producer = raw.get("ot_producer", {})
    _sha(producer.get("implementation_sha256"), "OT implementation")
    _require(producer.get("dtype") == "float64_cost_uniform_mass" and bool(producer.get("numpy"))
             and bool(producer.get("POT")), "OT numerical producer missing")
    _require(fixture or (raw.get("production_solver_used") is True and "UNIT_TEST" not in producer["POT"]),
             "Injected fixture solver is not production OT evidence")
    if binding.get("ot_producer") is not None:
        _require(producer == binding["ot_producer"], "OT producer differs from frozen backend identity")
    key = {name: raw[name] for name in ("schema_version", "encoding_ids", "numerical_contract", "solver", "ot_producer")}
    _require(raw.get("raw_pair_key") == digest(key), "Raw pair cache key mismatch")
    value = raw.get("distance")
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0
             and raw.get("distance_is_uncapped") is True, "Raw distance is missing/nonfinite/capped")


class _Records:
    def __init__(self, root: Path, spec: Mapping[str, Any], fixture: bool):
        self.root, self.spec, self.fixture = root, spec, fixture
        self.cache: dict[str, dict] = {}
        self.owners: dict[str, set[str]] = {}
        self.hashes: dict[str, str] = {}
        manifests = sorted((root / "producer_receipts").glob("*.jsonl"))
        _require(bool(manifests), "Missing real stage-producer record closures; raw values alone cannot pass")
        jobs = set()
        allowed_commits = set(spec["execution"].get("accepted_producer_commits", [])) | {spec["execution"]["execution_commit"]}
        _require(all(re.fullmatch(r"[0-9a-f]{40}", commit) for commit in allowed_commits), "Invalid explicit producer commit binding")
        for path in manifests:
            _require(not path.is_symlink(), "Symlinked producer journal")
            data = path.read_text()
            _require(not data or data.endswith("\n"), "Uncommitted partial producer journal line")
            for line in data.splitlines():
                row = json.loads(line)
                name = row.get("path", "")
                stages = _record_stages(name)
                if not stages:  # Audit/control outputs are not science inputs.
                    continue
                _require(row.get("science_hash") == spec["science_hash"]
                         and row.get("execution_commit") in allowed_commits, "Producer changed science/execution identity")
                _require(fixture or row.get("fixture") is not True, "Fixture evidence cannot pass production audit")
                job = str(row.get("job_id", ""))
                _require((fixture or job.isdecimal()) and int(row.get("pid", 0)) > 0 and row.get("created_at"),
                         "Missing genuine producer job/process identity")
                _require(fixture or (path.stem == job and job != os.environ.get("SLURM_JOB_ID")),
                         "Scientific producer cannot act as its own independent provenance auditor")
                _sha(row.get("sha256"), "producer artifact hash")
                if name not in self.hashes:
                    self.hashes[name] = row["sha256"]
                _require(self.hashes[name] == row["sha256"], f"Conflicting producer-bound record bytes: {name}")
                self.owners.setdefault(name, set()).update(stages)
                jobs.add(job)
        self.job_states = {} if fixture else _verify_closed_jobs(jobs)
        self.manifests = [str(p.relative_to(root)) for p in manifests]

    def get(self, relative: str, allowed_stages: set[str]) -> dict:
        _require(bool(self.owners.get(relative, set()) & allowed_stages), f"Missing stage-owned producer evidence: {relative}")
        if relative not in self.cache:
            # Do not even open a test scientific artifact before freeze checks.
            data = _path(self.root, relative).read_bytes()
            _require(hashlib.sha256(data).hexdigest() == self.hashes[relative],
                     f"Producer-bound record bytes changed: {relative}")
            self.cache[relative] = json.loads(data)
        row = self.cache[relative]
        _require(row.get("science_hash") == self.spec["science_hash"], f"Record changed science identity: {relative}")
        return row


def audit_bace_run(spec: Mapping[str, Any], root: str | Path, *, fixture: bool = False) -> dict[str, Any]:
    """Reconcile real BACE lineage in a fresh audit process, without new science.

    Missing evidence raises ProvenanceAuditError before any accepted receipt is
    returned. The caller writes a fresh audit result, never changes old records.
    """
    from src.baselines.cm_crem_experiment import load_parent_rows, science_identity, validate_contract
    validate_contract(dict(spec))
    _require(spec.get("science_hash") == science_identity(dict(spec)), "Resolved science identity drift")
    root = Path(root).resolve()
    _require(fixture or spec["campaign"]["primary"]["dataset"] == "bace", "This auditor is BACE-specific")
    db_path = spec["execution"].get("database_receipt")
    _require(bool(db_path) and Path(db_path).is_file(), "Official database receipt missing; generation is ASSET_BLOCKED")
    db = json.loads(Path(db_path).read_text())
    _require(db.get("status") == ("FIXTURE_STATIC_COPY" if fixture else "VERIFIED_STATIC_COPY")
             and db.get("url") == spec["upstream"]["database"]["url"], "Official database provenance is not verified")
    for key in ("compressed_sha256", "uncompressed_sha256"):
        _sha(db.get(key), key)
    records = _Records(root, spec, fixture)
    pilot = records.get("pilot/final_receipt.json", {"pilot-closeout"})
    _require(pilot.get("status") == "PILOT_ENGINEERING_PROTOCOL_ACCEPTED"
             and pilot.get("complete_end_to_end_pilot") is True and pilot.get("parent_count") == 32
             and pilot.get("real_nonself_wnode_pairs", 0) >= 64
             and pilot.get("scientific_parameters_tuned") is False,
             "No complete real 32-parent end-to-end pilot/cost admission")
    pool = records.get("pool_freeze.json", {"filter"})
    _require(pool.get("pool_sha256") == digest({k: v for k, v in pool.items() if k not in {"science_hash", "pool_sha256"}}),
             "Train pool content identity drift")
    freeze = SelectionFreeze.from_dict(records.get("selection_freeze.json", {"select"}))
    _require(freeze.frozen_pool_sha256 == pool["pool_sha256"], "Selection not bound to the original train pool")
    _require(freeze.theta == spec["resolved_evaluation"]["theta"]
             and freeze.cap == spec["resolved_evaluation"]["cap"], "Freeze changed original theta/cap")
    # Freeze verification precedes even audit-time loading of test parent data.
    parents = {split: load_parent_rows(dict(spec), split, root) for split in ("train", "calibration", "test")}
    _require(len(parents["train"]) == 386 and len(parents["calibration"]) == 66
             and len(parents["test"]) == 141, "Original BACE base cohorts changed")
    oracle = spec["resolved_oracle"]
    def prediction(row: Mapping[str, Any], parent: Mapping[str, Any] | None = None) -> None:
        _require(row.get("oracle_weight_sha256") == oracle.get("model_sha256")
                 and row.get("temperature") == oracle.get("temperature"), "Prediction changed original GINE/temperature")
        _require(type(row.get("predicted_label")) is int and row["predicted_label"] in (0, 1)
                 and bool(row.get("full_graph_id")), "Missing original binary GINE prediction/graph identity")
        if parent is not None:
            _require(row.get("parent_id") == parent["parent_id"] and row.get("smiles") == parent["smiles"],
                     "Prediction does not bind the authoritative parent")
    attr = records.get("attribution.json", {"attribution"})
    train_predictions = attr.get("train_predictions")
    _require(isinstance(train_predictions, list) and len(train_predictions) == len(parents["train"]),
             "Missing all-train prediction ledger; source subset/count cannot prove full cohort accounting")
    for row, parent in zip(train_predictions, parents["train"], strict=True):
        prediction(row, parent)
    eligible = [p["parent_id"] for p in train_predictions if p["predicted_label"] == 1]
    _require(attr["parents"] == [p for p in train_predictions if p["predicted_label"] == 1]
             and [r["parent_id"] for r in attr["records"]] == eligible,
             "Attribution/generation omitted or added a train-source parent")
    pilot_oracle = records.get("pilot/oracle.json", {"pilot-oracle"})
    _require(len(pilot_oracle.get("parents", [])) == 32
             and len({p["parent_id"] for p in pilot_oracle["parents"]}) == 32
             and {p["parent_id"] for p in pilot_oracle["parents"]} <= set(eligible),
             "Pilot is not 32 distinct original train-source parents")
    pilot_encoded = {r["candidate_id"]: r for r in pilot_oracle.get("encoded_graphs", [])}
    _require(set(pilot_encoded) == {p["full_graph_id"] for p in pilot_oracle["parents"]},
             "Pilot encodings do not bind the actual pilot parents")
    pilot_pairs = pilot_oracle.get("nonself_pairs", [])
    pair_ids = set()
    for raw in pilot_pairs:
        left, right = raw.get("parent_graph_id"), raw.get("prototype_graph_id")
        _require(left in pilot_encoded and right in pilot_encoded and left != right,
                 "Pilot benchmark is missing or is a self-pair")
        validate_raw_pair(raw, pilot_encoded[left], pilot_encoded[right], spec["resolved_wnode"], fixture=fixture)
        _require(math.isfinite(float(raw.get("seconds", -1))) and raw.get("seconds", -1) >= 0,
                 "Pilot raw pair lacks a real timing")
        pair_ids.add(tuple(sorted((left, right))))
    _require(len(pair_ids) >= 64 and pilot.get("real_nonself_wnode_pairs") == len(pilot_pairs),
             "Pilot real nonself count is a claim without 64 distinct raw graph pairs")
    accepted: dict[str, dict] = {}
    generation_statuses = Counter()
    for parent, attribution in zip(attr["parents"], attr["records"], strict=True):
        _require(attribution.get("input_smiles") == parent["smiles"] and attribution.get("split") == "train"
                 and attribution.get("oracle_weight_sha256") == oracle["model_sha256"]
                 and attribution.get("temperature_sha256") == oracle["temperature_sha256"],
                 "Attribution is not bound to original train SMILES/GINE/temperature")
        suffix = digest(parent["parent_id"])[:20] + ".json"
        generation = records.get("generation_units/"+suffix, {"generate", "pilot-generate"})
        validate_generation(generation, attribution, spec["science_hash"], fixture=fixture)
        _require(generation.get("database_uncompressed_sha256") == db["uncompressed_sha256"],
                 "Generation did not bind the adopted official database content")
        generation_statuses[generation["status"]] += 1
        filtered = records.get("filter_units/"+suffix, {"filter", "pilot-filter"})
        _require(filtered.get("status") == "FILTER_COMPLETE" and filtered.get("parent_id") == parent["parent_id"]
                 and filtered.get("test_loaded") is False and filtered.get("raw_count") == len(generation["retained_raw"]),
                 "Filter parent/funnel does not bind the generation terminal")
        prediction(filtered["source_prediction"], parent)
        _require(filtered["source_prediction"]["predicted_label"] == 1
                 and filtered.get("unique_target_count") == len(filtered.get("accepted", []))
                 and filtered.get("strict_flip_count") == sum(len(c.get("origins", [])) for c in filtered.get("accepted", []))
                 and 0 <= filtered["strict_flip_count"] <= filtered.get("chemically_valid_nonself_count", -1) <= filtered["raw_count"],
                 "Filter source/funnel counts do not reconcile")
        for candidate in filtered.get("accepted", []):
            prediction(candidate)
            _require(candidate["predicted_label"] == 0, "Non-flipping candidate entered train pool")
            cid = candidate["candidate_id"]
            identity = {k: candidate.get(k) for k in ("canonical_smiles", "graph_sha256", "feature_schema_sha256")}
            _require(cid == digest(identity) == candidate["full_graph_id"], "Candidate canonical graph identity changed")
            _require(cid != parent["full_graph_id"], "Unchanged parent entered the counterfactual pool")
            for origin in candidate.get("origins", []):
                index = origin.get("retained_raw_index")
                _require(origin.get("parent_id") == parent["parent_id"] and type(index) is int
                         and 0 <= index < len(generation["retained_raw"])
                         and origin.get("raw_id") == generation["retained_raw"][index]["raw_id"],
                         "Candidate origin is not an actually retained generated output")
            _require(bool(candidate.get("origins")), "Candidate has no train generation provenance")
            if cid in accepted:
                previous = accepted[cid]
                identity_fields = ("candidate_id", "canonical_smiles", "graph_sha256", "feature_schema_sha256",
                                   "oracle_weight_sha256", "temperature", "predicted_label", "full_graph_id")
                _require(all(candidate.get(k) == previous.get(k) for k in identity_fields),
                         "Conflicting canonical candidate identities")
                previous["origins"].extend(deepcopy(candidate["origins"]))
            else:
                accepted[cid] = deepcopy(candidate)
    ordered = sorted(accepted, key=lambda cid: (digest([spec["science_hash"], 7, cid]), cid))[:2000]
    _require(pool.get("candidate_ids") == ordered and pool.get("candidates") == [accepted[c] for c in ordered]
             and pool.get("retained_count") == len(ordered)
             and pool.get("unique_strict_flip_count_before_cap") == len(accepted)
             and pool.get("calibration_loaded") is False and pool.get("test_loaded") is False,
             "Frozen library is not the deterministic pre-calibration full train pool")
    encoded = records.get("pool_encodings.json", {"encode"})
    _require(encoded.get("pool_sha256") == pool["pool_sha256"], "Prototype encodings reference a different pool")
    embeddings = {r["candidate_id"]: r for r in encoded["records"]}
    _require(len(embeddings) == len(encoded["records"]) and set(embeddings) == set(ordered), "Missing/duplicate prototype encoding")
    for row in embeddings.values():
        validate_encoding(row, spec["resolved_wnode"])
    matrices = {}
    ot_producers = set()
    for split in ("calibration", "test"):
        cids = ordered if split == "calibration" else list(freeze.selected_candidate_ids)
        values, statuses, masks = [], [], []
        for parent in parents[split]:
            name = split+"/parents/"+digest(parent["parent_id"])[:20]+".json"
            row = records.get(name, {"calibrate" if split == "calibration" else "test"})
            prediction(row["prediction"], parent)
            source = row["prediction"]["predicted_label"] == 1
            _require(row.get("parent_id") == parent["parent_id"] and row.get("pool_sha256") == pool["pool_sha256"]
                     and [p["candidate_id"] for p in row["pairs"]] == cids, "Pair Cartesian identity/order differs")
            if source and cids:
                _require(isinstance(row.get("parent_encoding"), dict), "Missing saved parent encoding behind raw OT values")
                _require(row["parent_encoding"]["candidate_id"] == row["prediction"]["full_graph_id"], "Parent encoding is not the predicted graph")
            for pair in row["pairs"]:
                _require(pair.get("parent_id") == parent["parent_id"] and pair.get("pred_before") == row["prediction"]["predicted_label"]
                         and pair.get("pred_after") == 0 and pair.get("strict_flip") is source
                         and pair.get("kept_in_base_denominator") is True, "Strict-flip/source/base denominator changed")
                raw = pair.get("raw_distance")
                if source:
                    _require(isinstance(raw, dict), "Missing real raw OT distance for a strict-flip pair")
                    validate_raw_pair(raw, row["parent_encoding"], embeddings[pair["candidate_id"]], spec["resolved_wnode"], fixture=fixture)
                    _require(pair.get("distance") == raw["distance"] and pair.get("raw_pair_key") == raw["raw_pair_key"]
                             and pair.get("failure_reason") is None and pair.get("pair_status") == "OK", "Reduced distance differs from raw producer value")
                    ot_producers.add(digest(raw["ot_producer"]))
                else:
                    _require(raw is None and pair.get("distance") is None and pair.get("raw_pair_key") is None
                             and pair.get("pair_status") == "BEFORE_NOT_SOURCE" and pair.get("failure_reason") == "BEFORE_NOT_SOURCE",
                             "Non-source parent was silently computed/dropped/imputed")
            values.append([p["distance"] if source else math.inf for p in row["pairs"]])
            statuses.append([p["pair_status"] for p in row["pairs"]])
            masks.append(source)
        matrices[split] = (np.asarray(values, dtype=float).reshape(len(parents[split]), len(cids)),
            np.asarray(statuses).reshape(len(parents[split]), len(cids)), [p["parent_id"] for p in parents[split]], cids, masks)
    _require(len(ot_producers) <= 1, "Mixed unapproved OT producer backends within the experiment")
    cv, cs, cp, cc, cm = matrices["calibration"]
    replay = select_calibration(cv, pair_status=cs, parent_ids=cp, candidate_ids=cc, source_mask=cm,
        theta=freeze.theta, cap=freeze.cap, contract_sha256=spec["science_hash"], frozen_pool_sha256=pool["pool_sha256"])
    _require(replay.freeze_sha256 == freeze.freeze_sha256, "Saved calibration records do not reproduce the original freeze")
    tv, ts, tp, tc, tm = matrices["test"]
    evaluated = evaluate_frozen_test(freeze, tv, pair_status=ts, parent_ids=tp, candidate_ids=tc,
                                     source_mask=tm, contract_sha256=spec["science_hash"])
    result = {"schema_version": "cm_crem_bace_provenance_audit_v1",
        "status": "FIXTURE_PROVENANCE_VERIFIED" if fixture else "BACE_SAVED_RECORD_PROVENANCE_VERIFIED",
        "fixture": fixture, "science_hash": spec["science_hash"], "dataset": "bace", "oracle": "gine",
        "scientific_pass_claimed": False, "portable_acceptance_state": "PENDING_INDEPENDENT_SCIENTIFIC_SPOTCHECK",
        "first_missing_validation": "FIXED_HASH_FROZEN_TEST_RAW_SMILES_ORIGINAL_GINE_MOLCLR_EXACT_OT_RECOMPUTATION",
        "source_artifact_replay": False, "model_calls": 0, "ot_calls": 0, "new_generation_calls": 0,
        "train_parent_count": len(parents["train"]), "train_source_count": len(eligible),
        "calibration_parent_count": 66, "test_parent_count": 141,
        "generation_status_counts": dict(generation_statuses), "pool_candidate_count": len(ordered),
        "selected_candidate_count": len(freeze.selected_candidate_ids), "test_evaluation": evaluated.to_dict(),
        "database_receipt_sha256": file_sha(db_path), "producer_manifests": records.manifests,
        "producer_bound_records_sha256": digest(records.hashes), "bound_record_sha256": records.hashes,
        "producer_job_states": records.job_states,
        "auditor_pid": os.getpid(), "auditor_host": socket.gethostname()}
    result["provenance_sha256"] = digest(result)
    return result


def _same_numeric(actual: Any, expected: Any, role: str, *, atol: float = 0.0, rtol: float = 0.0) -> None:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    _require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all()
             and np.all(np.abs(a-b) <= atol + rtol*np.abs(b)), f"Independent recomputation mismatch: {role}")


def fixed_spotcheck_pairs(rows: Sequence[Mapping[str, Any]], selected_ids: Sequence[str],
                         science_hash: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Prespecified identity-only ordering; never rank by distance or outcome."""
    eligible = []
    seen = set()
    for row in rows:
        _require([p["candidate_id"] for p in row["pairs"]] == list(selected_ids), "Spotcheck escaped frozen test columns")
        for pair in row["pairs"]:
            key = (row["parent_id"], pair["candidate_id"])
            _require(key not in seen, "Duplicate frozen test pair")
            seen.add(key)
            if pair["pair_status"] == "OK":
                _require(pair.get("strict_flip") is True and pair.get("raw_distance") is not None,
                         "Nonlegal pair marked OK")
                if row["prediction"]["full_graph_id"] != pair["candidate_id"]:
                    eligible.append((row, pair))
    return sorted(eligible, key=lambda item: (
        digest([science_hash, "audit_spotcheck_v1", item[0]["parent_id"], item[1]["candidate_id"]]),
        item[0]["parent_id"], item[1]["candidate_id"]))[:8]


def independent_spotcheck(spec: Mapping[str, Any], root: str | Path, provenance: Mapping[str, Any], *,
                          fixture: bool = False, fixture_backend: Any = None) -> dict[str, Any]:
    """Recompute <=8 frozen test graph pairs in the existing independent audit job.

    Production owns the original model/backend constructors. An injected backend
    is test-only and can only produce a visibly FIXTURE receipt. No persistent
    encoding/pair cache is read or written; repeated OT is explicitly counted.
    This scientific spotcheck does not certify transfer or PDF visual QA.
    """
    from src.baselines.cm_crem_experiment import science_identity, validate_contract
    validate_contract(dict(spec))
    _require(spec["science_hash"] == science_identity(dict(spec)), "Spotcheck science identity drift")
    _require(provenance.get("provenance_sha256") == digest({k: v for k, v in provenance.items() if k != "provenance_sha256"})
             and provenance.get("science_hash") == spec["science_hash"]
             and provenance.get("fixture") is fixture
             and provenance.get("status") == ("FIXTURE_PROVENANCE_VERIFIED" if fixture else "BACE_SAVED_RECORD_PROVENANCE_VERIFIED"),
             "Spotcheck requires the completed matching saved-record provenance audit")
    _require(fixture or fixture_backend is None, "Injected spotcheck backend is forbidden in production")
    root = Path(root).resolve()
    hashes = provenance["bound_record_sha256"]
    _require(digest(hashes) == provenance["producer_bound_records_sha256"], "Provenance record inventory changed")
    def read(name: str) -> dict:
        path = _path(root, name)
        _require(name in hashes and file_sha(path) == hashes[name], f"Record changed after provenance audit: {name}")
        return json.loads(path.read_text())
    freeze = SelectionFreeze.from_dict(read("selection_freeze.json"))
    _require(freeze.contract_sha256 == spec["science_hash"]
             and freeze.theta == spec["resolved_evaluation"]["theta"] and freeze.cap == spec["resolved_evaluation"]["cap"],
             "Spotcheck freeze not bound to original numerical contract")
    evaluation_path = _path(root, "test_evaluation.json")
    _require(evaluation_path.is_file(), "Write reconciled test_evaluation.json before independent spotcheck")
    evaluation_bytes = evaluation_path.read_bytes()
    evaluation = PrefixEvaluation.from_dict(json.loads(evaluation_bytes))
    _require(evaluation.to_dict() == provenance["test_evaluation"], "Spotcheck final test evaluation differs from reconciled raw records")
    source_test_sha = hashlib.sha256(evaluation_bytes).hexdigest()
    pool = read("pool_freeze.json")
    _require(pool["pool_sha256"] == freeze.frozen_pool_sha256, "Spotcheck train library differs from freeze")
    # Only the already audited base test ledger is opened, after freeze check.
    names = sorted(name for name in hashes if name.startswith("test/parents/") and name.endswith(".json"))
    _require(len(names) == 141, "Spotcheck original BACE test ledger incomplete")
    test_rows = [read(name) for name in names]
    _require(len({r["parent_id"] for r in test_rows}) == 141, "Spotcheck duplicate test parent")
    sample = fixed_spotcheck_pairs(test_rows, freeze.selected_candidate_ids, spec["science_hash"])
    prototypes = {r["candidate_id"]: r for r in pool["candidates"]}
    saved_encodings = {r["candidate_id"]: r for r in read("pool_encodings.json")["records"]}
    oracle_binding = spec["resolved_oracle"]
    atol, rtol = float(oracle_binding.get("forward_atol", 0)), float(oracle_binding.get("forward_rtol", 0))
    _require(all(math.isfinite(x) and x >= 0 for x in (atol, rtol)), "Invalid original forward tolerance")
    if atol or rtol:
        _sha(oracle_binding.get("numerical_contract_sha256"), "existing original forward numerical contract")
    if fixture_backend is None:
        _require(not fixture, "Fixture audit must use an explicitly injected fixture backend")
        from src.baselines.cm_crem_oracle import FrozenCMOracle, FrozenCMWNode, raw_distance_record
        from src.baselines.cm_crem_runtime import require_compute_node
        require_compute_node()
        oracle = FrozenCMOracle.from_resolved(spec)
        wnode = FrozenCMWNode.from_resolved(spec)
        class Backend:
            def predict(self, rows):
                return oracle.predict_rows(rows, split="independent_audit")
            def encode(self, rows):
                return wnode.encode_rows(rows, featurizer=oracle.featurizer)
            def distance(self, left, right):
                return raw_distance_record(left, right, numerical_contract=spec["resolved_wnode"])
        backend = Backend()
    else:
        backend = fixture_backend
    oracle_count = encoding_count = 0
    def check_prediction(saved: Mapping[str, Any]) -> dict:
        nonlocal oracle_count
        # Strip saved outputs so a model adapter cannot accidentally echo them.
        inputs = {k: saved[k] for k in ("parent_id", "candidate_id", "smiles") if k in saved}
        fresh_rows = backend.predict([inputs])
        _require(len(fresh_rows) == 1, "Independent oracle omitted a sampled input")
        fresh = fresh_rows[0]
        oracle_count += 1
        for key in ("predicted_label", "full_graph_id", "oracle_weight_sha256", "temperature"):
            _require(fresh.get(key) == saved.get(key), f"Independent oracle identity/decision mismatch: {key}")
        for key in ("logits", "probabilities", "source_probability", "confidence"):
            _require(key in fresh and key in saved, f"Missing original prediction payload: {key}")
            _same_numeric(fresh[key], saved[key], "oracle."+key, atol=atol, rtol=rtol)
        return fresh
    def check_encoding(saved: Mapping[str, Any], smiles: str) -> dict:
        nonlocal encoding_count
        fresh_rows = backend.encode([{"smiles": smiles, "candidate_id": saved["candidate_id"]}])
        _require(len(fresh_rows) == 1, "Independent encoder omitted a sampled graph")
        fresh = fresh_rows[0]
        encoding_count += 1
        validate_encoding(fresh, spec["resolved_wnode"])
        # No new encoder/OT tolerance exists in this contract: exact comparison.
        _same_numeric(fresh["H"], saved["H"], "original MolCLR node H")
        for key in ("candidate_id", "atom_numbers", "producer", "node_extraction_version", "encoding_sha256"):
            _require(fresh.get(key) == saved.get(key), f"Independent encoding identity mismatch: {key}")
        return fresh
    checked = []
    for row, pair in sample:
        prototype = prototypes[pair["candidate_id"]]
        check_prediction(row["prediction"])
        check_prediction(prototype)
        left = check_encoding(row["parent_encoding"], row["prediction"]["smiles"])
        right = check_encoding(saved_encodings[pair["candidate_id"]], prototype["smiles"])
        fresh = backend.distance(left, right)
        validate_raw_pair(fresh, left, right, spec["resolved_wnode"], fixture=fixture)
        _require(fresh["ot_producer"] == pair["raw_distance"]["ot_producer"], "Independent OT producer changed")
        _same_numeric(fresh["distance"], pair["raw_distance"]["distance"], "raw exact WNode")
        checked.append({"parent_id": row["parent_id"], "candidate_id": pair["candidate_id"],
                        "raw_pair_key": fresh["raw_pair_key"], "distance": fresh["distance"]})
    zero_evidence = None
    if not sample:
        _require(not any(p["pair_status"] == "OK" for r in test_rows for p in r["pairs"]),
                 "No nonself sample cannot validate a nonzero legal-pair claim")
        _require(provenance["test_evaluation"].get("source_mask") is not None,
                 "Zero-pair claim lacks reconciled base evaluation")
        # Audit observed source and target labels, without any new test parents.
        train = read("attribution.json")["train_predictions"]
        sampled_predictions = []
        for cohort, rows in (("train", train), ("test", [r["prediction"] for r in test_rows])):
            for label in (0, 1):
                choices = sorted((r for r in rows if r["predicted_label"] == label),
                    key=lambda r: (digest([spec["science_hash"], "audit_zero_v1", cohort, r["parent_id"]]), r["parent_id"]))
                if choices:
                    check_prediction(choices[0])
                    sampled_predictions.append({"split": cohort, "parent_id": choices[0]["parent_id"], "label": label})
        for cid in sorted(freeze.selected_candidate_ids, key=lambda c: digest([spec["science_hash"], "audit_zero_target", c]))[:2]:
            check_prediction(prototypes[cid])
        _require(bool(sampled_predictions), "Empty audit sample cannot verify a zero run")
        _require(not freeze.selected_candidate_ids or not any(r["prediction"]["predicted_label"] == 1 for r in test_rows),
                 "Source parents and frozen targets exist but raw pair evidence is absent")
        zero_evidence = {"reason": "EMPTY_TRAIN_LIBRARY" if not freeze.selected_candidate_ids else "NO_TEST_SOURCE_PARENTS",
                         "sampled_predictions": sampled_predictions, "legal_pair_count": 0}
    if fixture:
        implementation_commit = "FIXTURE"
    else:
        implementation_commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
            "rev-parse", "HEAD"], check=True, text=True, capture_output=True, timeout=15).stdout.strip()
        _require(implementation_commit == spec["execution"]["execution_commit"], "Independent audit code commit differs from resolved execution")
    receipt = {"schema_version": "cm_crem_bace_independent_spotcheck_v1", "fixture": fixture,
        "status": "FIXTURE_SCIENTIFIC_SPOTCHECK_VERIFIED" if fixture else "CM_CREM_INDEPENDENT_SPOTCHECK_PASS",
        "independent": not fixture, "implementation_commit": implementation_commit,
        "implementation_sha256": file_sha(Path(__file__)), "checked_records_count": oracle_count + len(checked),
        "source_test_evaluation_sha256": source_test_sha,
        "source_selection_freeze_sha256": hashes["selection_freeze.json"],
        "science_hash": spec["science_hash"], "provenance_sha256": provenance["provenance_sha256"],
        "freeze_sha256": freeze.freeze_sha256, "sample_rule": "audit_spotcheck_v1_identity_hash_first_8",
        "checked_pairs": checked, "audit_spotcheck_count": len(checked), "oracle_graph_recomputations": oracle_count,
        "encoding_graph_recomputations": encoding_count, "cache_production_reuse_claimed": False,
        "new_generation_calls": 0, "new_test_parents_loaded": 0, "zero_evidence": zero_evidence,
        "oracle_atol": atol, "oracle_rtol": rtol, "encoding_and_distance_comparison": "EXACT",
        "portable_acceptance_state": "PENDING_EXPORT_VISUAL_QA_AND_PORTABLE_TRANSFER_VALIDATION",
        "scientific_pass_claimed": False, "auditor_pid": os.getpid(), "auditor_host": socket.gethostname()}
    receipt["spotcheck_sha256"] = digest(receipt)
    return receipt
